"""SD 1.5ベースモデルをローカルにダウンロードする。

初回のみオンライン接続が必要。以降は全工程オフラインで動作する。
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import PRETRAINED_MODEL_NAME, LOCAL_MODEL_DIR


def main():
    from diffusers import StableDiffusionPipeline

    output_dir = PROJECT_ROOT / LOCAL_MODEL_DIR
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"モデルは既にダウンロード済みです: {output_dir}")
        return

    print(f"モデルをダウンロード中: {PRETRAINED_MODEL_NAME}")
    print(f"保存先: {output_dir}")

    pipe = StableDiffusionPipeline.from_pretrained(
        PRETRAINED_MODEL_NAME,
        torch_dtype="auto",
    )
    pipe.save_pretrained(str(output_dir))
    print(f"ダウンロード完了: {output_dir}")


if __name__ == "__main__":
    main()
