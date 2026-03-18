"""SD 1.5 LoRA学習スクリプト。

peft (Parameter-Efficient Fine-Tuning) を使用してUNetのAttention層のみを学習する。
Full fine-tuneと比べてVRAM使用量が大幅に少なく（~3GB）、
成果物は小さなLoRAファイル（~150MB）で他モデルと組み合わせ可能。

使用方法:
    python -m accelerate.commands.launch train_lora.py
    python -m accelerate.commands.launch train_lora.py --resume outputs/lora_checkpoints/epoch_0002
"""

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from config import (
    PROJECT_ROOT,
    LOCAL_MODEL_DIR,
    PROCESSED_DATA_DIR,
    REG_DATA_DIR,
    LORA_CHECKPOINT_DIR,
    LORA_SAMPLE_DIR,
    LORA_LOG_DIR,
    LORA_RANK,
    LORA_ALPHA,
    LORA_LEARNING_RATE,
    LORA_LR_SCHEDULER,
    LORA_LR_WARMUP_STEPS,
    LORA_MAX_TRAIN_EPOCHS,
    LORA_TRAIN_BATCH_SIZE,
    LORA_GRADIENT_ACCUMULATION_STEPS,
    LORA_SAVE_EVERY_N_EPOCHS,
    LORA_OPTIMIZER,
    LORA_TARGET_MODULES,
    MIXED_PRECISION,
    GRADIENT_CHECKPOINTING,
    USE_XFORMERS,
    MAX_GRAD_NORM,
    ADAM_BETA1,
    ADAM_BETA2,
    ADAM_WEIGHT_DECAY,
    SAMPLE_EVERY_N_EPOCHS,
    SAMPLE_PROMPTS,
    SAMPLE_NEGATIVE_PROMPT,
    NUM_SAMPLE_IMAGES_PER_PROMPT,
    SAMPLE_NUM_INFERENCE_STEPS,
    SAMPLE_GUIDANCE_SCALE,
    SAMPLE_SEED,
    DATALOADER_NUM_WORKERS,
    REG_RATIO,
)
from dataset import BucketSampler, LatentDataset

logger = get_logger(__name__)

TRAINING_STATE_FILE = "training_state.json"


def parse_args():
    parser = argparse.ArgumentParser(description="SD 1.5 LoRA Training")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="再開するLoRAチェックポイントのパス (例: outputs/lora_checkpoints/epoch_0002)",
    )
    return parser.parse_args()


def create_optimizer(params, optimizer_name, lr, betas, weight_decay):
    """Optimizerを作成する。"""
    if optimizer_name == "adamw_8bit":
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(
                params, lr=lr, betas=betas, weight_decay=weight_decay
            )
        except ImportError:
            print("警告: bitsandbytesが見つかりません。通常のAdamWにフォールバックします。")
            optimizer_name = "adamw"

    return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)


def create_lr_scheduler(optimizer, scheduler_name, warmup_steps, total_steps):
    """学習率スケジューラを作成する。"""
    from diffusers.optimization import get_scheduler
    return get_scheduler(
        scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def convert_peft_to_kohya(peft_state_dict, lora_alpha, lora_rank):
    """peft形式のLoRA重みをKohya/ComfyUI互換形式に変換する。

    変換ルール:
    - base_model.model. プレフィックスを除去
    - .lora_A.weight → .lora_down.weight
    - .lora_B.weight → .lora_up.weight
    - ドットをアンダースコアに変換（最後の .weight は除く）
    - 各LoRA層に対応するalpha値テンソルを追加
    """
    kohya_dict = {}
    processed_keys = set()

    for key, value in peft_state_dict.items():
        # peftのLoRAキーのみ処理
        if "lora_A" not in key and "lora_B" not in key:
            continue

        # base_model.model. プレフィックスを除去
        new_key = key
        if new_key.startswith("base_model.model."):
            new_key = new_key[len("base_model.model."):]

        # lora_A → lora_down, lora_B → lora_up
        new_key = new_key.replace(".lora_A.weight", ".lora_down.weight")
        new_key = new_key.replace(".lora_B.weight", ".lora_up.weight")

        # ドットをアンダースコアに変換（.weightの前まで）
        parts = new_key.rsplit(".weight", 1)
        converted = parts[0].replace(".", "_") + ".weight"

        # lora_sd_ プレフィックスを付与（ComfyUI Load LoRA互換）
        kohya_dict[converted] = value

        # alpha値を追加（lora_downキーに対応するalphaを一度だけ追加）
        if "lora_down" in converted:
            alpha_key = converted.replace("lora_down.weight", "alpha")
            if alpha_key not in processed_keys:
                kohya_dict[alpha_key] = torch.tensor(float(lora_alpha))
                processed_keys.add(alpha_key)

    return kohya_dict


def save_lora_checkpoint(unet, optimizer, lr_scheduler, epoch, global_step, output_path: Path):
    """LoRAチェックポイントを保存する。"""
    output_path.mkdir(parents=True, exist_ok=True)

    # peft形式で保存（再開用）
    unet.save_pretrained(str(output_path / "peft_model"))

    # Kohya/ComfyUI互換形式で保存
    peft_state_dict = unet.state_dict()
    kohya_dict = convert_peft_to_kohya(peft_state_dict, LORA_ALPHA, LORA_RANK)
    save_file(kohya_dict, str(output_path / "lora.safetensors"))

    # Optimizer + Scheduler + 学習状態を保存
    torch.save(optimizer.state_dict(), str(output_path / "optimizer.pt"))
    torch.save(lr_scheduler.state_dict(), str(output_path / "lr_scheduler.pt"))

    training_state = {
        "epoch": epoch,
        "global_step": global_step,
    }
    with open(output_path / TRAINING_STATE_FILE, "w") as f:
        json.dump(training_state, f)

    print(f"LoRAチェックポイント保存: {output_path} (epoch={epoch}, step={global_step})")
    size_mb = (output_path / "lora.safetensors").stat().st_size / (1024 * 1024)
    print(f"  ComfyUI用: lora.safetensors ({size_mb:.1f} MB)")


def generate_samples(
    vae, text_encoder, unet, tokenizer, noise_scheduler,
    epoch: int, global_step: int, output_dir: Path, writer: SummaryWriter,
):
    """学習中にサンプル画像を生成する。"""
    sample_vae = copy.deepcopy(vae).to("cuda", dtype=torch.float32)

    pipeline = StableDiffusionPipeline(
        vae=sample_vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipeline.set_progress_bar_config(disable=True)
    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    sample_dir = output_dir / f"epoch_{epoch:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device="cuda").manual_seed(SAMPLE_SEED)

    for i, prompt in enumerate(SAMPLE_PROMPTS):
        for j in range(NUM_SAMPLE_IMAGES_PER_PROMPT):
            with torch.no_grad():
                image = pipeline(
                    prompt=prompt,
                    negative_prompt=SAMPLE_NEGATIVE_PROMPT if prompt else "",
                    num_inference_steps=SAMPLE_NUM_INFERENCE_STEPS,
                    guidance_scale=SAMPLE_GUIDANCE_SCALE if prompt else 1.0,
                    generator=generator,
                ).images[0]

            filename = f"prompt{i:02d}_{j:02d}.png"
            image.save(sample_dir / filename)

            import torchvision.transforms.functional as TF
            img_tensor = TF.to_tensor(image)
            writer.add_image(
                f"samples/prompt_{i}", img_tensor, global_step=global_step
            )

    del sample_vae, pipeline
    torch.cuda.empty_cache()
    print(f"サンプル画像生成: {sample_dir}")


def main():
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=LORA_GRADIENT_ACCUMULATION_STEPS,
        mixed_precision=MIXED_PRECISION,
    )

    set_seed(42)

    model_path = str(PROJECT_ROOT / LOCAL_MODEL_DIR)

    # === 再開チェックポイントの解決 ===
    resume_path = None
    start_epoch = 0
    start_global_step = 0

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = PROJECT_ROOT / resume_path
        if not resume_path.exists():
            print(f"エラー: チェックポイントが見つかりません: {resume_path}")
            sys.exit(1)
        state_file = resume_path / TRAINING_STATE_FILE
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            start_epoch = state["epoch"]
            start_global_step = state["global_step"]
            print(f"再開: epoch {start_epoch}, step {start_global_step} から")
    else:
        # 自動検出
        ckpt_base = PROJECT_ROOT / LORA_CHECKPOINT_DIR
        if ckpt_base.exists():
            ckpts = sorted([
                d for d in ckpt_base.iterdir()
                if d.is_dir() and (d / TRAINING_STATE_FILE).exists()
            ])
            if ckpts:
                resume_path = ckpts[-1]
                with open(resume_path / TRAINING_STATE_FILE) as f:
                    state = json.load(f)
                start_epoch = state["epoch"]
                start_global_step = state["global_step"]
                print(f"最新チェックポイントを自動検出: {resume_path.name}")
                print(f"再開: epoch {start_epoch}, step {start_global_step} から")

    # === モデル読み込み ===
    print("モデルを読み込み中...")
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")

    # 全パラメータを凍結
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    # LoRA適用
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.0,
    )
    unet = get_peft_model(unet, lora_config)

    # 再開時はpeft重みを復元
    if resume_path and (resume_path / "peft_model").exists():
        from peft import set_peft_model_state_dict
        import safetensors.torch
        peft_path = resume_path / "peft_model"
        adapter_path = peft_path / "adapter_model.safetensors"
        if adapter_path.exists():
            adapter_weights = safetensors.torch.load_file(str(adapter_path))
            set_peft_model_state_dict(unet, adapter_weights)
            print(f"LoRA重みを復元: {peft_path}")

    # 学習可能パラメータ数を表示
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    print(f"LoRAパラメータ: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Gradient checkpointing
    if GRADIENT_CHECKPOINTING:
        unet.enable_gradient_checkpointing()

    # xformers / SDPA
    if USE_XFORMERS:
        try:
            unet.enable_xformers_memory_efficient_attention()
            print("xformers有効化")
        except Exception:
            print("xformers無効 → PyTorch SDPAにフォールバック")

    # fp16に変換（凍結モデル）
    vae.to(dtype=torch.float16)
    text_encoder.to(dtype=torch.float16)

    # === データセット ===
    print("データセットを読み込み中...")
    processed_dir = PROJECT_ROOT / PROCESSED_DATA_DIR
    reg_dir = PROJECT_ROOT / REG_DATA_DIR

    if not processed_dir.exists() or not any(processed_dir.glob("*.pt")):
        print("エラー: キャッシュ済みlatentが見つかりません。")
        print("先に以下を実行してください:")
        print("  python scripts/01_prepare_images.py")
        print("  python scripts/03_cache_latents.py")
        sys.exit(1)

    dataset = LatentDataset(
        data_dir=processed_dir,
        reg_dir=reg_dir if reg_dir.exists() else None,
        tokenizer=tokenizer,
        reg_ratio=REG_RATIO,
    )

    sampler = BucketSampler(dataset, batch_size=LORA_TRAIN_BATCH_SIZE, shuffle=True)

    dataloader = DataLoader(
        dataset,
        batch_size=LORA_TRAIN_BATCH_SIZE,
        sampler=sampler,
        num_workers=DATALOADER_NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # === Optimizer & Scheduler ===
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = create_optimizer(
        lora_params, LORA_OPTIMIZER, LORA_LEARNING_RATE,
        (ADAM_BETA1, ADAM_BETA2), ADAM_WEIGHT_DECAY,
    )

    total_steps = math.ceil(len(dataloader) / LORA_GRADIENT_ACCUMULATION_STEPS) * LORA_MAX_TRAIN_EPOCHS
    lr_scheduler = create_lr_scheduler(
        optimizer, LORA_LR_SCHEDULER, LORA_LR_WARMUP_STEPS, total_steps,
    )

    # === Accelerate準備 ===
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )

    # Optimizer/Scheduler状態復元
    if resume_path:
        opt_path = resume_path / "optimizer.pt"
        sched_path = resume_path / "lr_scheduler.pt"
        if opt_path.exists():
            print("Optimizer状態を復元中...")
            optimizer.load_state_dict(torch.load(str(opt_path), map_location="cpu", weights_only=True))
        if sched_path.exists():
            print("LR Scheduler状態を復元中...")
            lr_scheduler.load_state_dict(torch.load(str(sched_path), map_location="cpu", weights_only=True))

    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)

    # === TensorBoard ===
    log_dir = PROJECT_ROOT / LORA_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    # === 学習ループ ===
    print(f"\nLoRA学習開始:")
    print(f"  学習画像: {len(dataset.train_items)}枚")
    print(f"  正則化画像: {len(dataset.reg_items)}枚")
    print(f"  合計: {len(dataset)}枚")
    print(f"  バッチサイズ: {LORA_TRAIN_BATCH_SIZE} × 勾配蓄積 {LORA_GRADIENT_ACCUMULATION_STEPS}")
    print(f"  エポック: {LORA_MAX_TRAIN_EPOCHS}")
    print(f"  総ステップ: {total_steps}")
    print(f"  学習率: {LORA_LEARNING_RATE}")
    print(f"  LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}")
    print(f"  Optimizer: {LORA_OPTIMIZER}")
    if start_epoch > 0:
        print(f"  再開: epoch {start_epoch + 1}から (step {start_global_step})")
    print()

    global_step = start_global_step

    for epoch in range(start_epoch, LORA_MAX_TRAIN_EPOCHS):
        unet.train()
        epoch_loss = 0.0
        num_batches = 0

        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{LORA_MAX_TRAIN_EPOCHS}")

        for batch in progress:
            with accelerator.accumulate(unet):
                latents = batch["latent"].to(dtype=torch.float16)
                input_ids = batch["input_ids"]

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids)[0]

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                    dtype=torch.long,
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                noise_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), MAX_GRAD_NORM)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1

            epoch_loss += loss.detach().item()
            num_batches += 1

            current_lr = lr_scheduler.get_last_lr()[0]
            progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/lr", current_lr, global_step)

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1} 完了 - 平均Loss: {avg_loss:.4f}")
        writer.add_scalar("train/epoch_loss", avg_loss, epoch + 1)

        # チェックポイント保存
        if (epoch + 1) % LORA_SAVE_EVERY_N_EPOCHS == 0 or (epoch + 1) == LORA_MAX_TRAIN_EPOCHS:
            ckpt_dir = PROJECT_ROOT / LORA_CHECKPOINT_DIR / f"epoch_{epoch + 1:04d}"
            unwrapped_unet = accelerator.unwrap_model(unet)
            save_lora_checkpoint(
                unwrapped_unet, optimizer, lr_scheduler,
                epoch + 1, global_step, ckpt_dir,
            )

        # サンプル画像生成
        if (epoch + 1) % SAMPLE_EVERY_N_EPOCHS == 0:
            unwrapped_unet = accelerator.unwrap_model(unet)
            sample_output = PROJECT_ROOT / LORA_SAMPLE_DIR
            generate_samples(
                vae, text_encoder, unwrapped_unet, tokenizer, noise_scheduler,
                epoch + 1, global_step, sample_output, writer,
            )

    writer.close()
    print("\nLoRA学習完了!")
    print(f"チェックポイント: {PROJECT_ROOT / LORA_CHECKPOINT_DIR}")
    print(f"サンプル画像: {PROJECT_ROOT / LORA_SAMPLE_DIR}")
    print(f"TensorBoardログ: {PROJECT_ROOT / LORA_LOG_DIR}")
    print(f"\nComfyUIで使用するには:")
    print(f"  各チェックポイント内の lora.safetensors をComfyUIの models/loras/ にコピー")


if __name__ == "__main__":
    main()
