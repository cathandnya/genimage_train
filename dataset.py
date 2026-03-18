"""学習用Datasetクラス。

キャッシュ済みlatent + キャプション + 正則化画像インターリーブ + バケットサンプラー対応。
"""

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler
from transformers import CLIPTokenizer

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


class LatentDataset(Dataset):
    """キャッシュ済みlatentとキャプションを読み込むDataset。"""

    def __init__(
        self,
        data_dir: str | Path,
        reg_dir: str | Path | None,
        tokenizer: CLIPTokenizer,
        reg_ratio: float = 1.0,
    ):
        self.tokenizer = tokenizer
        self.train_items = self._scan_dir(Path(data_dir))

        self.reg_items = []
        if reg_dir and Path(reg_dir).exists():
            self.reg_items = self._scan_dir(Path(reg_dir))

        # 正則化画像を学習画像に対してreg_ratio分だけ追加
        if self.reg_items and reg_ratio > 0:
            num_reg = int(len(self.train_items) * reg_ratio)
            # 正則化画像をリピートして必要数に合わせる
            reg_repeated = self.reg_items * (num_reg // len(self.reg_items) + 1)
            self.reg_items = reg_repeated[:num_reg]

        self.items = self.train_items + self.reg_items
        random.shuffle(self.items)

    def _scan_dir(self, directory: Path) -> list[dict]:
        """ディレクトリからlatent(.pt)とキャプション(.txt)のペアを検索する。"""
        items = []
        for pt_path in sorted(directory.glob("*.pt")):
            caption_path = pt_path.with_suffix(".txt")
            caption = ""
            if caption_path.exists():
                caption = caption_path.read_text(encoding="utf-8").strip()
            items.append(
                {
                    "latent_path": pt_path,
                    "caption": caption,
                }
            )
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # latent読み込み
        data = torch.load(item["latent_path"], map_location="cpu", weights_only=True)
        latent = data["latent"]  # shape: (4, H//8, W//8)

        # キャプショントークナイズ
        tokens = self.tokenizer(
            item["caption"],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "latent": latent,
            "input_ids": tokens.input_ids.squeeze(0),
        }


class BucketSampler(Sampler):
    """同じバケット（解像度）の画像をまとめてバッチにするサンプラー。

    バケッティング使用時、同一バッチ内は同一解像度になる。
    """

    def __init__(self, dataset: LatentDataset, batch_size: int = 1, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # バケット別にインデックスを分類
        self.buckets: dict[tuple[int, int], list[int]] = {}
        for idx, item in enumerate(dataset.items):
            data = torch.load(item["latent_path"], map_location="cpu", weights_only=True)
            latent = data["latent"]
            # latent shape: (4, H//8, W//8) → 解像度を復元
            h = latent.shape[1] * 8
            w = latent.shape[2] * 8
            key = (w, h)
            if key not in self.buckets:
                self.buckets[key] = []
            self.buckets[key].append(idx)

    def __iter__(self):
        # バケットごとにbatch_size単位のグループを作成
        # 端数はバッチ境界をまたがないよう最後のグループに含める
        bucket_groups = []
        for indices in self.buckets.values():
            if self.shuffle:
                random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                bucket_groups.append(indices[i : i + self.batch_size])

        if self.shuffle:
            random.shuffle(bucket_groups)

        all_indices = [idx for group in bucket_groups for idx in group]
        return iter(all_indices)

    def __len__(self):
        return len(self.dataset)
