"""正則化画像をベースSD 1.5モデルから生成する。

Catastrophic forgettingを防止するために、ベースモデルで
クラス画像を生成して data/reg/ に保存する。
"""

import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    PROJECT_ROOT as ROOT,
    LOCAL_MODEL_DIR,
    REG_DATA_DIR,
    REG_NUM_IMAGES,
    REG_PROMPTS,
    REG_NEGATIVE_PROMPT,
    REG_NUM_INFERENCE_STEPS,
    REG_GUIDANCE_SCALE,
    RESOLUTION,
    BUCKETS,
    USE_BUCKETING,
)


def main():
    from diffusers import StableDiffusionPipeline

    model_path = ROOT / LOCAL_MODEL_DIR
    if not model_path.exists():
        print(f"エラー: モデルが見つかりません: {model_path}")
        print("先に python scripts/00_download_model.py を実行してください。")
        sys.exit(1)

    reg_dir = ROOT / REG_DATA_DIR
    reg_dir.mkdir(parents=True, exist_ok=True)

    # 既存画像をカウント
    existing = list(reg_dir.glob("*.png"))
    if len(existing) >= REG_NUM_IMAGES:
        print(f"正則化画像は既に{len(existing)}枚あります（目標: {REG_NUM_IMAGES}枚）。")
        return

    remaining = REG_NUM_IMAGES - len(existing)
    start_idx = len(existing)

    print(f"正則化画像を生成します: {remaining}枚")
    print(f"モデル: {model_path}")

    pipe = StableDiffusionPipeline.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cuda")

    # メモリ最適化
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pipe.enable_attention_slicing()

    # バケットサイズのリストを用意（ランダムに使用）
    if USE_BUCKETING:
        sizes = BUCKETS
    else:
        sizes = [(RESOLUTION, RESOLUTION)]

    generator = torch.Generator(device="cuda")

    for i in tqdm(range(remaining), desc="正則化画像生成中"):
        idx = start_idx + i
        prompt = REG_PROMPTS[idx % len(REG_PROMPTS)]
        size = sizes[idx % len(sizes)]

        generator.manual_seed(idx)

        image = pipe(
            prompt=prompt,
            negative_prompt=REG_NEGATIVE_PROMPT,
            width=size[0],
            height=size[1],
            num_inference_steps=REG_NUM_INFERENCE_STEPS,
            guidance_scale=REG_GUIDANCE_SCALE,
            generator=generator,
        ).images[0]

        img_path = reg_dir / f"reg_{idx:05d}.png"
        image.save(img_path, "PNG")

        # 空キャプション
        caption_path = reg_dir / f"reg_{idx:05d}.txt"
        caption_path.write_text("", encoding="utf-8")

    del pipe
    torch.cuda.empty_cache()

    total = len(list(reg_dir.glob("*.png")))
    print(f"\n完了: 正則化画像 {total}枚 ({reg_dir})")


if __name__ == "__main__":
    main()
