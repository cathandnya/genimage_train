"""LoRA学習済みモデルで画像を生成する推論スクリプト。

ベースモデルにLoRA重みを適用して推論する。

使用方法:
    python scripts/06_inference_lora.py --prompt "nystyle, a woman standing" --lora outputs/lora_checkpoints/epoch_0010/lora.safetensors
    python scripts/06_inference_lora.py --prompt "nystyle, a woman in a red dress" --lora outputs/lora_checkpoints/epoch_0010/lora.safetensors --num 4 --seed 42
    python scripts/06_inference_lora.py --prompt "nystyle, a woman" --lora outputs/lora_checkpoints/epoch_0010/lora.safetensors --scale 0.8
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="SD 1.5 LoRA 推論")
    parser.add_argument(
        "--prompt", type=str, required=True, help="生成プロンプト"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="low quality, blurry, deformed",
        help="ネガティブプロンプト",
    )
    parser.add_argument(
        "--lora",
        type=str,
        required=True,
        help="LoRAファイルのパス (例: outputs/lora_checkpoints/epoch_0010/lora.safetensors)",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="LoRA適用強度 (デフォルト: 1.0)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="出力ディレクトリ"
    )
    parser.add_argument("--num", type=int, default=1, help="生成枚数")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    parser.add_argument("--steps", type=int, default=30, help="推論ステップ数")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--width", type=int, default=512, help="画像の幅")
    parser.add_argument("--height", type=int, default=512, help="画像の高さ")
    return parser.parse_args()


def load_kohya_lora(pipe, lora_path, scale=1.0):
    """Kohya形式のLoRA重みをdiffusersパイプラインに適用する。"""
    from safetensors.torch import load_file

    state_dict = load_file(str(lora_path))

    # diffusersのload_lora_weightsを使用
    # Kohya形式は自動変換される
    pipe.load_lora_weights(
        str(Path(lora_path).parent),
        weight_name=Path(lora_path).name,
    )

    # LoRA強度の調整
    if scale != 1.0:
        pipe.fuse_lora(lora_scale=scale)


def main():
    from diffusers import StableDiffusionPipeline
    from config import LOCAL_MODEL_DIR

    args = parse_args()

    lora_path = Path(args.lora)
    if not lora_path.is_absolute():
        lora_path = PROJECT_ROOT / lora_path

    if not lora_path.exists():
        print(f"エラー: LoRAファイルが見つかりません: {lora_path}")
        sys.exit(1)

    model_path = str(PROJECT_ROOT / LOCAL_MODEL_DIR)
    print(f"ベースモデルを読み込み中: {model_path}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cuda")

    print(f"LoRAを適用中: {lora_path} (scale={args.scale})")
    load_kohya_lora(pipe, lora_path, scale=args.scale)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pipe.enable_attention_slicing()

    # 出力先
    output_dir = Path(args.output) if args.output else lora_path.parent / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    print(f"プロンプト: {args.prompt}")
    print(f"生成枚数: {args.num}")
    print(f"出力先: {output_dir}")

    for i in range(args.num):
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
        ).images[0]

        filename = f"gen_{i:04d}.png"
        image.save(output_dir / filename)
        print(f"  保存: {filename}")

    print(f"\n完了: {args.num}枚の画像を生成しました → {output_dir}")


if __name__ == "__main__":
    main()
