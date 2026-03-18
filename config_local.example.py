"""ユーザーローカル設定のサンプル。

このファイルを config_local.py にコピーして、変更したい値だけ記述してください。
config_local.py は .gitignore に含まれるため git に影響しません。

使い方:
    cp config_local.example.py config_local.py
    # config_local.py を編集
"""

# === キャプション ===
# COMMON_CAPTION = "nystyle"

# === Full Fine-Tune ===
# LEARNING_RATE = 1e-6
# MAX_TRAIN_EPOCHS = 10
# TRAIN_BATCH_SIZE = 1
# GRADIENT_ACCUMULATION_STEPS = 4

# === LoRA ===
# LORA_RANK = 128
# LORA_ALPHA = 128
# LORA_LEARNING_RATE = 1e-4
# LORA_TRAIN_BATCH_SIZE = 4
# LORA_GRADIENT_ACCUMULATION_STEPS = 2
# LORA_MAX_TRAIN_EPOCHS = 10
# LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]
# LORA_TRAIN_TEXT_ENCODER = True
# LORA_TEXT_ENCODER_LR = 5e-5

# === サンプリング ===
# SAMPLE_PROMPTS = [
#     "nystyle, a woman standing in a garden",
#     "nystyle, portrait of a woman, soft lighting",
# ]
