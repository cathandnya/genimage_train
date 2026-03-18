"""学習画像の前処理: アスペクト比維持リサイズ + バケッティング + キャプション生成。

data/raw/ の画像を読み込み、アスペクト比を維持したままリサイズして
最適なバケットに割り当て、data/processed/ に保存する。
クロップは行わないため全身画像が切れない。
"""

import sys
from pathlib import Path
from collections import defaultdict

from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    PROJECT_ROOT as ROOT,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    BUCKETS,
    USE_BUCKETING,
    RESOLUTION,
    COMMON_CAPTION,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def find_best_bucket(width: int, height: int) -> tuple[int, int]:
    """画像のアスペクト比に最も近いバケットを選択する。"""
    aspect = width / height
    best_bucket = None
    best_diff = float("inf")
    for bw, bh in BUCKETS:
        bucket_aspect = bw / bh
        diff = abs(aspect - bucket_aspect)
        if diff < best_diff:
            best_diff = diff
            best_bucket = (bw, bh)
    return best_bucket


def resize_to_bucket(img: Image.Image, bucket_w: int, bucket_h: int) -> Image.Image:
    """画像をバケットサイズにリサイズする。

    アスペクト比を維持してバケットに収まるようリサイズし、
    余白を黒でパディングする。
    """
    w, h = img.size
    scale = min(bucket_w / w, bucket_h / h)
    new_w = round(w * scale)
    new_h = round(h * scale)

    img = img.resize((new_w, new_h), Image.LANCZOS)

    # パディング（中央配置）
    padded = Image.new("RGB", (bucket_w, bucket_h), (0, 0, 0))
    offset_x = (bucket_w - new_w) // 2
    offset_y = (bucket_h - new_h) // 2
    padded.paste(img, (offset_x, offset_y))

    return padded


def resize_simple(img: Image.Image, resolution: int) -> Image.Image:
    """バケッティングなしの場合: 長辺をresolutionにリサイズ + パディング。"""
    w, h = img.size
    scale = min(resolution / w, resolution / h)
    new_w = round(w * scale)
    new_h = round(h * scale)

    img = img.resize((new_w, new_h), Image.LANCZOS)

    padded = Image.new("RGB", (resolution, resolution), (0, 0, 0))
    offset_x = (resolution - new_w) // 2
    offset_y = (resolution - new_h) // 2
    padded.paste(img, (offset_x, offset_y))

    return padded


def main():
    raw_dir = ROOT / RAW_DATA_DIR
    out_dir = ROOT / PROCESSED_DATA_DIR

    if not raw_dir.exists():
        print(f"エラー: {raw_dir} が見つかりません。学習画像を配置してください。")
        sys.exit(1)

    image_files = [
        f for f in sorted(raw_dir.iterdir()) if f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        print(f"エラー: {raw_dir} に画像が見つかりません。")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    bucket_counts = defaultdict(int)
    processed = 0
    errors = 0

    for img_path in tqdm(image_files, desc="前処理中"):
        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            if USE_BUCKETING:
                bucket = find_best_bucket(w, h)
                result = resize_to_bucket(img, *bucket)
                bucket_counts[bucket] += 1
            else:
                result = resize_simple(img, RESOLUTION)
                bucket_counts[(RESOLUTION, RESOLUTION)] += 1

            # 保存
            out_path = out_dir / f"{img_path.stem}.png"
            result.save(out_path, "PNG")

            # キャプションファイル（COMMON_CAPTIONの値を書き込む）
            caption_path = out_dir / f"{img_path.stem}.txt"
            caption_path.write_text(COMMON_CAPTION, encoding="utf-8")

            processed += 1

        except Exception as e:
            print(f"警告: {img_path.name} の処理に失敗: {e}")
            errors += 1

    # サマリー
    print(f"\n処理完了: {processed}枚 (エラー: {errors}枚)")
    print(f"出力先: {out_dir}")
    if USE_BUCKETING:
        print("\nバケット分布:")
        for bucket, count in sorted(bucket_counts.items()):
            print(f"  {bucket[0]}x{bucket[1]}: {count}枚")


if __name__ == "__main__":
    main()
