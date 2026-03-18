# SD 1.5 Fine-Tune / LoRA

Stable Diffusion 1.5をfine-tuneして、特定のリアル系/写真風スタイルの人体表現を学習させるプロジェクト。

2つの学習方式に対応:
- **Full Fine-Tune**: UNet全体を学習。最高品質だがVRAM ~11GB、成果物 ~2GB
- **LoRA**: Attention層のみを学習。VRAM ~3GB、成果物 ~150MB、他モデルと組み合わせ可能

成果物は`.safetensors`で、ComfyUIでそのまま使用可能。

## 前提条件

- Python 3.12+
- conda (Miniconda / Anaconda)
- NVIDIA GPU 12GB+ VRAM
- CUDA 12.x
- 約10GBの空きディスク（モデル + データ + チェックポイント）

## 環境構築

```bash
# 1. conda環境を作成
conda create -n genimg python=3.12 -y
conda activate genimg

# 2. PyTorchをCUDA対応でインストール
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3. その他の依存パッケージをインストール
pip install -r requirements.txt

# 4. ベースモデルをダウンロード（初回のみ、約4GB）
python scripts/00_download_model.py
```

> **Note**: xformersのインストールに失敗する場合はスキップしても動作します（PyTorch組み込みのSDPAが自動的にフォールバックします）。

## 使い方

> **重要**: 以下の全コマンドは `conda activate genimg` を実行してから行ってください。

### Full Fine-Tune vs LoRA

どちらの学習方式でもStep 1〜4は共通です。Step 5以降が分岐します。

| 項目 | Full Fine-Tune | LoRA |
|---|---|---|
| VRAM使用量 | ~11GB | ~3GB |
| 成果物サイズ | ~2GB | ~150MB |
| 学習速度 | 遅い | 速い |
| 品質 | 最高 | 高（rankで調整可能） |
| 他モデルとの組み合わせ | 不可（モデルそのもの） | 可能（他LoRA・他モデルに適用） |
| ComfyUI | Load Checkpoint | Load LoRA |

### Step 1: 学習データの配置

`data/raw/` に学習用画像を配置します（50〜200枚推奨、1000枚以上も可）。

対応形式: `.png`, `.jpg`, `.jpeg`, `.webp`

### Step 2: データ前処理

```bash
python scripts/01_prepare_images.py
```

画像をアスペクト比を維持したままリサイズし、バケッティングで分類します。全身が切れないようクロップは行いません。各画像にキャプションファイル（.txt）を生成します。

### Step 3: 正則化画像生成（推奨）

```bash
python scripts/02_generate_reg.py
```

ベースモデルからクラス画像を生成します。catastrophic forgetting（壊滅的忘却）を防止するために重要です。

### Step 4: Latentキャッシュ

```bash
python scripts/03_cache_latents.py
```

全画像をVAEでエンコードし、latentを事前キャッシュします。学習時のVRAM使用量を約400MB削減します。

---

### Step 5A: Full Fine-Tune学習

```bash
python -m accelerate.commands.launch train.py
```

別ターミナルでTensorBoardを起動してモニタリングできます:

```bash
tensorboard --logdir outputs/logs
```

### Step 6A: Full Fine-Tune推論テスト

```bash
python scripts/04_inference.py --prompt "nystyle, a woman standing" --checkpoint outputs/checkpoints/epoch_0010
```

### Step 7A: Full Fine-Tune → ComfyUI用に変換

```bash
# 単一 .safetensors に変換
python scripts/05_convert_checkpoint.py --checkpoint outputs/checkpoints/epoch_0010

# fp16で保存（ファイルサイズ約半分、品質への影響は軽微）
python scripts/05_convert_checkpoint.py --checkpoint outputs/checkpoints/epoch_0010 --half
```

変換後の `model.safetensors` をComfyUIの `models/checkpoints/` にコピーし、**Load Checkpoint** ノードで選択。

---

### Step 5B: LoRA学習

```bash
python -m accelerate.commands.launch train_lora.py
```

TensorBoardでモニタリング:

```bash
tensorboard --logdir outputs/lora_logs
```

### Step 6B: LoRA推論テスト

```bash
# 基本
python scripts/06_inference_lora.py --prompt "nystyle, a woman standing" --lora outputs/lora_checkpoints/epoch_0010/lora.safetensors

# LoRA強度を調整（0.0〜1.0、デフォルト1.0）
python scripts/06_inference_lora.py --prompt "nystyle, a woman standing" --lora outputs/lora_checkpoints/epoch_0010/lora.safetensors --scale 0.8

# 複数枚生成
python scripts/06_inference_lora.py --prompt "nystyle, a woman in a red dress" --lora outputs/lora_checkpoints/epoch_0010/lora.safetensors --num 4
```

### Step 7B: LoRA → ComfyUIで使用

LoRAは変換不要です。チェックポイント内の `lora.safetensors` をそのままComfyUIにコピーします。

1. `outputs/lora_checkpoints/epoch_XXXX/lora.safetensors` をComfyUIの `models/loras/` にコピー
2. ComfyUIで **Load LoRA** ノードを追加し、コピーしたLoRAを選択
3. ベースモデル（SD 1.5）の **Load Checkpoint** と組み合わせて使用
4. Load LoRAノードの `strength_model` でLoRA強度を調整可能

---

### 一括実行

```bash
# Full Fine-Tune（Step 1〜5を順番に実行）
python run_all.py

# LoRA（Step 1〜3 + LoRA学習・推論を順番に実行）
python run_all.py --lora

# 途中から再開
python run_all.py --from 4
python run_all.py --lora --from 4

# 特定ステップをスキップ
python run_all.py --skip 2 5
```

## パラメータ調整

`config.py`にデフォルト値が定義されています。変更したい場合は`config_local.py`を作成して上書きしてください。`config_local.py`はgit管理外なので`git pull`で上書きされません。

```bash
# サンプルをコピーして編集
cp config_local.example.py config_local.py
```

`config_local.py`には変更したい値だけ記述すればOKです:

```python
# config_local.py の例
COMMON_CAPTION = "nystyle"
LORA_RANK = 256
LORA_TRAIN_BATCH_SIZE = 4
```

### Full Fine-Tune パラメータ

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `LEARNING_RATE` | `1e-6` | 学習率。full fine-tuneでは低めが安全。範囲: `5e-7`〜`2e-6` |
| `MAX_TRAIN_EPOCHS` | `10` | 学習エポック数。100枚なら10〜20が目安 |
| `TRAIN_BATCH_SIZE` | `1` | バッチサイズ。12GB VRAMでは1固定 |
| `GRADIENT_ACCUMULATION_STEPS` | `4` | 勾配蓄積。実効バッチサイズ = BATCH × この値 |
| `RESOLUTION` | `512` | ベース解像度。バケッティングで可変 |
| `SAVE_EVERY_N_EPOCHS` | `2` | チェックポイント保存間隔 |

### LoRA パラメータ

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `LORA_RANK` | `128` | LoRAのランク。高いほど表現力↑、ファイルサイズ↑。範囲: `4`〜`256` |
| `LORA_ALPHA` | `128` | スケーリング係数。通常はrankと同じ値 |
| `LORA_LEARNING_RATE` | `1e-4` | 学習率。LoRAはfull fine-tuneより高め。範囲: `1e-5`〜`5e-4` |
| `LORA_TRAIN_BATCH_SIZE` | `2` | バッチサイズ。VRAM使用量が少ないのでバッチ2が可能 |
| `LORA_GRADIENT_ACCUMULATION_STEPS` | `2` | 勾配蓄積。実効バッチサイズ = 2 × 2 = 4 |
| `LORA_MAX_TRAIN_EPOCHS` | `10` | 学習エポック数 |
| `LORA_SAVE_EVERY_N_EPOCHS` | `2` | チェックポイント保存間隔 |
| `LORA_TARGET_MODULES` | `["to_q","to_k","to_v","to_out.0"]` | 学習対象のAttention層 |

### VRAM不足時の対処

**Full Fine-Tune:**
1. `RESOLUTION`を`448`に下げる
2. `GRADIENT_ACCUMULATION_STEPS`を`8`に上げて`TRAIN_BATCH_SIZE`を確認（1であること）
3. `USE_XFORMERS`が`True`であることを確認
4. `CACHE_LATENTS`が`True`であることを確認

**LoRA:** VRAM ~3GBなので通常は問題になりません。それでもOOMの場合は`LORA_TRAIN_BATCH_SIZE`を`1`に下げてください。

## キャプション（トリガーワード）を変えて再学習

`config.py`の`COMMON_CAPTION`で全学習画像に共通のキャプションを設定できます。

| 設定 | 効果 |
|---|---|
| `COMMON_CAPTION = ""` | unconditional学習。スタイルがデフォルト出力になる |
| `COMMON_CAPTION = "nystyle"` | トリガーワード付き学習。推論時に`nystyle, <プロンプト>`でスタイル適用 |

キャプションを変更して再学習する手順:

```bash
# 1. config.pyのCOMMON_CAPTIONを編集
#    例: COMMON_CAPTION = "nystyle"

# 2. キャプションファイルを再生成（画像のリサイズは既存を上書き）
python scripts/01_prepare_images.py

# 3. latent再キャッシュは不要（キャプションは.txtから都度読み込むため）

# 4. 既存チェックポイントを削除（前回の学習状態から再開させないため）
#    Full Fine-Tune: outputs/checkpoints/ の中身を手動で削除、またはリネーム
#    LoRA: outputs/lora_checkpoints/ の中身を手動で削除、またはリネーム

# 5. 学習開始
python -m accelerate.commands.launch train.py        # Full Fine-Tune
python -m accelerate.commands.launch train_lora.py   # LoRA
```

> **Note**: 正則化画像のキャプションは常に空です（再生成不要）。トリガーワードを正則化画像に付けるとスタイル学習の効果が打ち消されます。

## ComfyUIでの使用方法

### Full Fine-Tune の場合

1. 変換スクリプトでComfyUI互換の単一ファイルを生成:
   ```bash
   python scripts/05_convert_checkpoint.py --checkpoint outputs/checkpoints/epoch_0010 --half
   ```
2. 生成された `model.safetensors` をComfyUIの `models/checkpoints/` にコピー
3. ComfyUIを起動（または再読み込み）
4. **Load Checkpoint** ノードでコピーしたモデルを選択

SD 1.5互換のチェックポイントなので、既存のSD 1.5用ワークフローがそのまま使えます。

### LoRA の場合

1. `outputs/lora_checkpoints/epoch_XXXX/lora.safetensors` をComfyUIの `models/loras/` にコピー
2. ComfyUIを起動（または再読み込み）
3. ワークフローに **Load LoRA** ノードを追加
4. **Load Checkpoint**（SD 1.5ベースモデル）→ **Load LoRA** → **KSampler** の順に接続
5. Load LoRAノードの `strength_model` で適用強度を調整（0.0〜1.0）

LoRAは他のLoRAやモデルと組み合わせて使えます。Stability Matrixの場合は `E:\StabilityMatrix\Data\Models\Lora\` に配置してください。

## プロジェクト構成

```
genimg/
├── README.md                  # このファイル
├── requirements.txt           # 依存パッケージ
├── config.py                  # ハイパーパラメータ
├── dataset.py                 # Datasetクラス（共通）
├── train.py                   # Full Fine-Tune学習スクリプト
├── train_lora.py              # LoRA学習スクリプト
├── scripts/
│   ├── 00_download_model.py   # モデルダウンロード
│   ├── 01_prepare_images.py   # データ前処理
│   ├── 02_generate_reg.py     # 正則化画像生成
│   ├── 03_cache_latents.py    # Latentキャッシュ
│   ├── 04_inference.py        # Full Fine-Tune推論
│   ├── 05_convert_checkpoint.py # ComfyUI用変換（Full Fine-Tune用）
│   └── 06_inference_lora.py   # LoRA推論
├── models/                    # ベースモデル（ローカル）
├── data/
│   ├── raw/                   # 元画像
│   ├── processed/             # 前処理済み画像
│   └── reg/                   # 正則化画像
└── outputs/
    ├── checkpoints/           # Full Fine-Tune成果物
    ├── lora_checkpoints/      # LoRA成果物 (lora.safetensors)
    ├── samples/               # Full Fine-Tuneサンプル画像
    ├── lora_samples/          # LoRAサンプル画像
    ├── logs/                  # Full Fine-Tune TensorBoardログ
    └── lora_logs/             # LoRA TensorBoardログ
```

## トラブルシューティング

### CUDA Out of Memory (OOM)

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

`config.py`で以下を確認:
- `TRAIN_BATCH_SIZE = 1`
- `USE_XFORMERS = True`
- `GRADIENT_CHECKPOINTING = True`
- `CACHE_LATENTS = True`

それでも解決しない場合は`RESOLUTION = 448`に下げてください。

### bitsandbytes (Windows)

```
RuntimeError: bitsandbytes was not compiled with GPU support
```

最新版をインストール:
```bash
pip install bitsandbytes>=0.45.0
```

### xformers

```
ImportError: cannot import name 'xformers'
```

xformersが未インストールまたは非対応の場合、自動的にPyTorch SDPAにフォールバックします。動作に問題はありません。明示的にインストールする場合:
```bash
pip install xformers
```

### 過学習の兆候

- 生成画像がすべて似通っている
- プロンプトの指示が効かない
- 学習データと同じ画像が出力される

対策: `MAX_TRAIN_EPOCHS`を減らす、または早いエポックのチェックポイントを使用してください。

## オフライン動作

初回の`00_download_model.py`と`pip install`のみオンライン接続が必要です。モデルダウンロード後は全工程（前処理・学習・推論）がオフラインで動作します。

モデルはHuggingFaceキャッシュではなくプロジェクト内の`models/`に保存されるため、ポータブルです。
