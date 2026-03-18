"""Step 1〜5を順番に実行するワンショットスクリプト。

使用方法:
    conda activate genimg
    python run_all.py              # Full Fine-Tune
    python run_all.py --lora       # LoRA学習

各ステップの途中で失敗した場合はそこで停止する。
--skip オプションで特定ステップをスキップできる。
--from オプションで途中のステップから再開できる。
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

STEPS = [
    {
        "num": 1,
        "name": "データ前処理",
        "script": "scripts/01_prepare_images.py",
        "description": "画像リサイズ + バケッティング + キャプション生成",
        "check": lambda: any((PROJECT_ROOT / "data" / "processed").glob("*.png")),
    },
    {
        "num": 2,
        "name": "正則化画像生成",
        "script": "scripts/02_generate_reg.py",
        "description": "ベースモデルからクラス画像を生成（catastrophic forgetting防止）",
        "check": lambda: any((PROJECT_ROOT / "data" / "reg").glob("*.png")),
    },
    {
        "num": 3,
        "name": "Latentキャッシュ",
        "script": "scripts/03_cache_latents.py",
        "description": "VAEで全画像をエンコード → .pt保存",
        "check": lambda: any((PROJECT_ROOT / "data" / "processed").glob("*.pt")),
    },
    {
        "num": 4,
        "name": "学習",
        "script": "train.py",
        "description": "SD 1.5 UNet full fine-tune",
        "use_accelerate": True,
        "check": lambda: any((PROJECT_ROOT / "outputs" / "checkpoints").iterdir())
        if (PROJECT_ROOT / "outputs" / "checkpoints").exists()
        else False,
    },
    {
        "num": 5,
        "name": "推論テスト",
        "script": "scripts/04_inference.py",
        "description": "学習済みモデルでサンプル画像を生成",
        "extra_args": lambda: find_inference_args(),
        "check": lambda: True,
    },
]


LORA_STEPS = [
    {
        "num": 4,
        "name": "LoRA学習",
        "script": "train_lora.py",
        "description": "SD 1.5 UNet LoRA学習",
        "use_accelerate": True,
        "check": lambda: any((PROJECT_ROOT / "outputs" / "lora_checkpoints").iterdir())
        if (PROJECT_ROOT / "outputs" / "lora_checkpoints").exists()
        else False,
    },
    {
        "num": 5,
        "name": "LoRA推論テスト",
        "script": "scripts/06_inference_lora.py",
        "description": "LoRA適用でサンプル画像を生成",
        "extra_args": lambda: find_lora_inference_args(),
        "check": lambda: True,
    },
]


def find_inference_args():
    """最新のチェックポイントを探して推論引数を返す。"""
    ckpt_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    if not ckpt_dir.exists():
        return []
    checkpoints = sorted(ckpt_dir.iterdir())
    if not checkpoints:
        return []
    latest = checkpoints[-1]
    return [
        "--prompt", "a woman standing, full body",
        "--checkpoint", str(latest),
        "--num", "2",
    ]


def find_lora_inference_args():
    """最新のLoRAチェックポイントを探して推論引数を返す。"""
    ckpt_dir = PROJECT_ROOT / "outputs" / "lora_checkpoints"
    if not ckpt_dir.exists():
        return []
    checkpoints = sorted(ckpt_dir.iterdir())
    if not checkpoints:
        return []
    latest = checkpoints[-1]
    lora_file = latest / "lora.safetensors"
    if not lora_file.exists():
        return []
    return [
        "--prompt", "nystyle, a woman standing, full body",
        "--lora", str(lora_file),
        "--num", "2",
    ]


def run_step(step, dry_run=False):
    """1つのステップを実行する。"""
    print(f"\n{'='*60}")
    print(f"Step {step['num']}: {step['name']}")
    print(f"  {step['description']}")
    print(f"{'='*60}\n")

    if dry_run:
        print("  [DRY RUN] スキップ")
        return True

    script_path = PROJECT_ROOT / step["script"]

    if step.get("use_accelerate"):
        cmd = [sys.executable, "-m", "accelerate.commands.launch", str(script_path)]
    else:
        cmd = [sys.executable, str(script_path)]

    extra_args_fn = step.get("extra_args")
    if extra_args_fn:
        extra = extra_args_fn()
        cmd.extend(extra)

    start = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - start

    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    if result.returncode != 0:
        print(f"\n[エラー] Step {step['num']} が失敗しました (コード: {result.returncode})")
        print(f"  経過時間: {minutes}分{seconds}秒")
        return False

    print(f"\n[完了] Step {step['num']}: {step['name']} ({minutes}分{seconds}秒)")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="SD 1.5 Fine-Tune 一括実行")
    parser.add_argument(
        "--from", type=int, default=1, dest="from_step",
        help="開始ステップ番号 (デフォルト: 1)",
    )
    parser.add_argument(
        "--skip", type=int, nargs="*", default=[],
        help="スキップするステップ番号 (例: --skip 2 5)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="実行せずにステップ一覧を表示",
    )
    parser.add_argument(
        "--lora", action="store_true",
        help="LoRA学習モードで実行（Step 4-5がLoRA用に変わる）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 前提チェック
    raw_dir = PROJECT_ROOT / "data" / "raw"
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        print("エラー: data/raw/ に学習画像を配置してください。")
        sys.exit(1)

    model_dir = PROJECT_ROOT / "models" / "stable-diffusion-v1-5"
    if not model_dir.exists() or not any(model_dir.iterdir()):
        print("エラー: ベースモデルが見つかりません。")
        print("先に実行: python scripts/00_download_model.py")
        sys.exit(1)

    # LoRAモードの場合、Step 4-5を差し替え
    steps = STEPS
    if args.lora:
        steps = [s for s in STEPS if s["num"] <= 3] + LORA_STEPS

    mode = "LoRA" if args.lora else "Full Fine-Tune"
    print(f"SD 1.5 {mode} パイプライン")
    print(f"  開始ステップ: {args.from_step}")
    print(f"  スキップ: {args.skip if args.skip else 'なし'}")

    total_start = time.time()
    failed = False

    for step in steps:
        if step["num"] < args.from_step:
            continue
        if step["num"] in args.skip:
            print(f"\n[スキップ] Step {step['num']}: {step['name']}")
            continue

        success = run_step(step, dry_run=args.dry_run)
        if not success:
            failed = True
            print(f"\nStep {step['num']} で停止しました。")
            print(f"修正後に --from {step['num']} で再開できます。")
            break

    total_elapsed = time.time() - total_start
    total_min = int(total_elapsed // 60)
    total_sec = int(total_elapsed % 60)

    if not failed:
        print(f"\n{'='*60}")
        print(f"全ステップ完了! (合計: {total_min}分{total_sec}秒)")
        print(f"{'='*60}")
        if args.lora:
            print(f"\nLoRAチェックポイント: outputs/lora_checkpoints/")
            print(f"サンプル画像: outputs/lora_samples/")
            print(f"ComfyUIで使う場合: 各epoch内の lora.safetensors を")
            print(f"  ComfyUIの models/loras/ にコピーしてください。")
        else:
            print(f"\nチェックポイント: outputs/checkpoints/")
            print(f"サンプル画像: outputs/samples/")
            print(f"ComfyUIで使う場合: outputs/checkpoints/ 内の .safetensors を")
            print(f"  ComfyUIの models/checkpoints/ にコピーしてください。")


if __name__ == "__main__":
    main()
