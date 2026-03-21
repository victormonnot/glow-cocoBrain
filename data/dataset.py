"""
dataset.py — Dataset COCO Captions pour GLoW
==============================================
Charge les features ResNet-50 pré-extraites + captions tokenisées.

Chaque sample retourne:
  {"vision": (2048,) features, "text": (max_seq_len,) token IDs}

Les paires sont aplaties : 118k images × 5 captions = ~590k paires train.
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict

from data.tokenizer import SimpleTokenizer


class COCODataset(Dataset):
    """Dataset COCO Captions avec features pré-extraites."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        max_seq_len: int = 32,
    ):
        self.max_seq_len = max_seq_len

        # Load tokenizer
        vocab_path = os.path.join(data_root, "vocab.json")
        self.tokenizer = SimpleTokenizer.load(vocab_path)

        # Load pre-extracted features
        features_path = os.path.join(data_root, f"features_{split}.npy")
        ids_path = os.path.join(data_root, f"image_ids_{split}.json")

        self.features = np.load(features_path)  # (N_images, 2048)
        with open(ids_path) as f:
            image_ids = json.load(f)
        # Mapping image_id → index in features array
        self.id_to_idx = {img_id: idx for idx, img_id in enumerate(image_ids)}

        # Load annotations and build (image_id, caption) pairs
        ann_path = os.path.join(
            data_root, "annotations", f"captions_{split}2017.json"
        )
        with open(ann_path) as f:
            annotations = json.load(f)

        self.samples = [
            (ann["image_id"], ann["caption"])
            for ann in annotations["annotations"]
            if ann["image_id"] in self.id_to_idx
        ]

        print(
            f"COCODataset({split}): {len(self.samples)} pairs, "
            f"{len(self.id_to_idx)} images, "
            f"vocab={self.tokenizer.vocab_size}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_id, caption = self.samples[idx]

        # Vision: pre-extracted ResNet-50 features
        feat_idx = self.id_to_idx[image_id]
        vision = torch.tensor(self.features[feat_idx], dtype=torch.float32)

        # Text: tokenized caption
        text = self.tokenizer.encode(caption, self.max_seq_len)

        return {"vision": vision, "text": text}


class DummyMultimodalDataset(Dataset):
    """Dataset de test avec données aléatoires (pour debug)."""

    def __init__(
        self,
        num_samples: int = 10000,
        feature_dim: int = 2048,
        vocab_size: int = 10000,
        seq_len: int = 32,
    ):
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(idx)
        return {
            "vision": torch.rand(self.feature_dim, generator=gen),
            "text": torch.randint(0, self.vocab_size, (self.seq_len,), generator=gen),
        }


def get_dataset(config: dict, split: str = "train") -> Dataset:
    """Factory : retourne le bon dataset selon la config."""
    name = config["data"]["dataset"]

    if name == "dummy":
        return DummyMultimodalDataset(
            feature_dim=config["vision"]["feature_dim"],
            vocab_size=config["text"]["vocab_size"],
            seq_len=config["text"]["max_seq_len"],
        )
    elif name == "coco":
        return COCODataset(
            data_root=config["data"]["data_root"],
            split=split,
            max_seq_len=config["text"]["max_seq_len"],
        )
    else:
        raise ValueError(
            f"Dataset '{name}' pas implémenté. Utilise 'dummy' ou 'coco'."
        )
