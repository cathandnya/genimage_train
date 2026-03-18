"""全画像をVAEでエンコードしてlatentを事前キャッシュする。

学習時にVAEをGPUに載せる必要がなくなり、約400MBのVRAMを節約する。
"""

import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    PROJECT_ROOT as ROOT,
    LOCAL_MODEL_DIR,
    PROCESSED_DATA_DIR,
    REG_DATA_DIR,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def encode_images(vae, image_dir: Path, device: torch.device):
    """ディレクトリ内の全画像をVAEでエンコードして.ptファイルとして保存する。"""
    image_files = [
        f for f in sorted(image_dir.iterdir()) if f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        print(f"  スキップ: {image_dir} に画像がありません")
        return 0

    cached = 0
    for img_path in tqdm(image_files, desc=f"  {image_dir.name}"):
        pt_path = img_path.with_suffix(".pt")
        if pt_path.exists():
            cached += 1
            continue

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # 画像をテンソルに変換 [-1, 1]
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        pixel_values = transform(img).unsqueeze(0).to(device, dtype=torch.float16)

        # VAEエンコード
        with torch.no_grad():
            latent = vae.encode(pixel_values).latent_dist.sample()
            latent = latent * vae.config.scaling_factor

        # CPUに戻して保存（サイズとlatent）
        torch.save(
            {
                "latent": latent.squeeze(0).cpu(),
                "original_size": (w, h),
            },
            pt_path,
        )
        cached += 1

    return cached


def main():
    from diffusers import AutoencoderKL

    model_path = ROOT / LOCAL_MODEL_DIR
    if not model_path.exists():
        print(f"エラー: モデルが見つかりません: {model_path}")
        print("先に python scripts/00_download_model.py を実行してください。")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("VAEを読み込み中...")
    vae = AutoencoderKL.from_pretrained(
        str(model_path), subfolder="vae", torch_dtype=torch.float16
    )
    vae = vae.to(device)
    vae.eval()

    total = 0

    # 学習画像
    processed_dir = ROOT / PROCESSED_DATA_DIR
    if processed_dir.exists():
        print(f"\n学習画像のlatentをキャッシュ中: {processed_dir}")
        total += encode_images(vae, processed_dir, device)

    # 正則化画像
    reg_dir = ROOT / REG_DATA_DIR
    if reg_dir.exists():
        print(f"\n正則化画像のlatentをキャッシュ中: {reg_dir}")
        total += encode_images(vae, reg_dir, device)

    del vae
    torch.cuda.empty_cache()

    print(f"\n完了: {total}枚のlatentをキャッシュしました")


if __name__ == "__main__":
    main()
