"""学習済みモデルで画像を生成する推論スクリプト。

使用方法:
    python scripts/04_inference.py --prompt "a woman standing" --checkpoint outputs/checkpoints/epoch_0010
    python scripts/04_inference.py --prompt "a woman in a red dress" --checkpoint outputs/checkpoints/epoch_0010 --num 4 --seed 42
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="SD 1.5 Fine-Tuned Model 推論")
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
        "--checkpoint",
        type=str,
        required=True,
        help="チェックポイントのパス (例: outputs/checkpoints/epoch_0010)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="出力ディレクトリ (デフォルト: チェックポイント内)"
    )
    parser.add_argument("--num", type=int, default=1, help="生成枚数")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    parser.add_argument("--steps", type=int, default=30, help="推論ステップ数")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--width", type=int, default=512, help="画像の幅")
    parser.add_argument("--height", type=int, default=512, help="画像の高さ")
    return parser.parse_args()


def main():
    from diffusers import StableDiffusionPipeline

    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path

    if not ckpt_path.exists():
        print(f"エラー: チェックポイントが見つかりません: {ckpt_path}")
        sys.exit(1)

    print(f"モデルを読み込み中: {ckpt_path}")
    pipe = StableDiffusionPipeline.from_pretrained(
        str(ckpt_path),
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cuda")

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pipe.enable_attention_slicing()

    # 出力先
    output_dir = Path(args.output) if args.output else ckpt_path / "inference"
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
