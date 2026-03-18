"""全ハイパーパラメータと設定を一括管理する。"""

from pathlib import Path

# === パス ===
PROJECT_ROOT = Path(__file__).resolve().parent
PRETRAINED_MODEL_NAME = "stable-diffusion-v1-5/stable-diffusion-v1-5"
LOCAL_MODEL_DIR = "models/stable-diffusion-v1-5"

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
REG_DATA_DIR = "data/reg"
OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = "outputs/checkpoints"
SAMPLE_DIR = "outputs/samples"
LOG_DIR = "outputs/logs"

# === 画像前処理 ===
RESOLUTION = 512
USE_BUCKETING = True
# バケット定義: (幅, 高さ) — 面積が約262144px (512*512) になる組み合わせ
BUCKETS = [
    (512, 512),
    (448, 576),
    (576, 448),
    (384, 640),
    (640, 384),
    (320, 768),
    (768, 320),
]

# === キャプション ===
# 全学習画像に共通で付与するキャプション（トリガーワード）
# 空文字列 = unconditional学習（スタイルがデフォルト出力になる）
# 例: "nystyle" → 推論時に "nystyle, a woman in a red dress" でスタイル適用
COMMON_CAPTION = "nystyle"

# === 正則化画像 ===
REG_NUM_IMAGES = 200
REG_PROMPTS = [
    "a photo of a person",
    "a person standing",
    "full body photo of a person",
    "a person, plain background",
]
REG_NEGATIVE_PROMPT = "low quality, blurry, deformed"
REG_NUM_INFERENCE_STEPS = 30
REG_GUIDANCE_SCALE = 7.5

# === 学習 ===
TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-6
LR_SCHEDULER = "constant_with_warmup"
LR_WARMUP_STEPS = 500
OPTIMIZER = "adamw_8bit"  # "adamw_8bit", "adamw", "lion"
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
MAX_TRAIN_EPOCHS = 10
MIXED_PRECISION = "fp16"
GRADIENT_CHECKPOINTING = True
USE_XFORMERS = True
CACHE_LATENTS = True
FREEZE_TEXT_ENCODER = True

# === 保存・サンプリング ===
SAVE_EVERY_N_EPOCHS = 2
SAMPLE_EVERY_N_EPOCHS = 1
SAMPLE_PROMPTS = [
    "a woman standing in a garden",
    "portrait of a woman, soft lighting",
    "a woman in a red dress, full body",
    "",  # unconditional
]
SAMPLE_NEGATIVE_PROMPT = "low quality, blurry, deformed"
NUM_SAMPLE_IMAGES_PER_PROMPT = 1
SAMPLE_NUM_INFERENCE_STEPS = 30
SAMPLE_GUIDANCE_SCALE = 7.5
SAMPLE_SEED = 42

# === LoRA ===
LORA_RANK = 128
LORA_ALPHA = 128  # alpha == rank → スケーリング係数1.0
LORA_LEARNING_RATE = 1e-4  # LoRAはfull fine-tuneより高い学習率が使える
LORA_LR_SCHEDULER = "cosine"
LORA_LR_WARMUP_STEPS = 100
LORA_MAX_TRAIN_EPOCHS = 10
LORA_TRAIN_BATCH_SIZE = 2  # LoRAはVRAM使用量が少ないためバッチ2が可能
LORA_GRADIENT_ACCUMULATION_STEPS = 2  # 実効バッチサイズ = 2 × 2 = 4
LORA_SAVE_EVERY_N_EPOCHS = 2
LORA_OPTIMIZER = "adamw_8bit"
LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]  # UNet Attention層
LORA_TRAIN_TEXT_ENCODER = True  # Text EncoderにもLoRAを適用（プロンプト追従性向上）
LORA_TEXT_ENCODER_LR = 5e-5  # Text Encoder用学習率（UNetより低め推奨）
LORA_TEXT_ENCODER_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]  # CLIP Attention層

# LoRA用チェックポイント・サンプル出力先
LORA_CHECKPOINT_DIR = "outputs/lora_checkpoints"
LORA_SAMPLE_DIR = "outputs/lora_samples"
LORA_LOG_DIR = "outputs/lora_logs"

# === データローダー ===
DATALOADER_NUM_WORKERS = 0  # Windows環境では0推奨
REG_RATIO = 1.0  # 正則化画像の比率（1.0 = 学習画像と同数）

# === ユーザーローカル設定の読み込み ===
# config_local.py が存在する場合、上記の値を上書きする
# config_local.py は .gitignore に含まれるため、各環境固有の設定を安全に保持できる
try:
    from config_local import *  # noqa: F401, F403
except ImportError:
    pass
