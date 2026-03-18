"""diffusers形式のチェックポイントをComfyUI互換の単一.safetensorsに変換する。

diffusers公式の変換スクリプトをダウンロードして使用するため、
キーマッピングの正確性が保証される。

使用方法:
    python scripts/05_convert_checkpoint.py --checkpoint outputs/checkpoints/epoch_0010
    python scripts/05_convert_checkpoint.py --checkpoint outputs/checkpoints/epoch_0010 --output my_model.safetensors
    python scripts/05_convert_checkpoint.py --checkpoint outputs/checkpoints/epoch_0010 --half
"""

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONVERT_SCRIPT_URL = "https://raw.githubusercontent.com/huggingface/diffusers/v0.31.0/scripts/convert_diffusers_to_original_stable_diffusion.py"
CONVERT_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "_convert_diffusers_to_original.py"


def ensure_convert_script():
    """公式変換スクリプトをダウンロードする（初回のみ）。"""
    if CONVERT_SCRIPT_PATH.exists():
        return

    print(f"公式変換スクリプトをダウンロード中...")
    print(f"  URL: {CONVERT_SCRIPT_URL}")
    urllib.request.urlretrieve(CONVERT_SCRIPT_URL, str(CONVERT_SCRIPT_PATH))
    print(f"  保存先: {CONVERT_SCRIPT_PATH}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="diffusers → ComfyUI互換 .safetensors 変換"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="diffusers形式チェックポイントのパス (例: outputs/checkpoints/epoch_0010)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="出力ファイルパス (デフォルト: チェックポイントディレクトリ内に model.safetensors)",
    )
    parser.add_argument(
        "--half", action="store_true",
        help="fp16で保存（ファイルサイズ半減、品質への影響は軽微）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path

    if not ckpt_path.exists():
        print(f"エラー: チェックポイントが見つかりません: {ckpt_path}")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
    else:
        output_path = ckpt_path / "model.safetensors"

    # 公式スクリプトを取得
    ensure_convert_script()

    # 公式スクリプトを実行
    cmd = [
        sys.executable,
        str(CONVERT_SCRIPT_PATH),
        "--model_path", str(ckpt_path),
        "--checkpoint_path", str(output_path),
        "--use_safetensors",
    ]

    if args.half:
        cmd.append("--half")

    print(f"\n変換開始:")
    print(f"  入力: {ckpt_path}")
    print(f"  出力: {output_path}")
    print()

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nエラー: 変換に失敗しました (コード: {result.returncode})")
        sys.exit(1)

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n完了: {output_path} ({size_mb:.0f} MB)")
        print(f"ComfyUIの models/checkpoints/ にコピーして使用してください。")


if __name__ == "__main__":
    main()
