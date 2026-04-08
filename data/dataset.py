"""
dataset.py — Dataset COCO Captions pour GLoW
==============================================
Charge les features ResNet-50 pré-extraites + embeddings E5-large-v2 pré-extraits.

Chaque sample retourne:
  {"vision": (2048,) features, "text": (1024,) E5 embeddings,
   "image_id": int, "is_paired": bool}

Les paires sont aplaties : 118k images × 5 captions = ~590k paires train.
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Dict


class COCODataset(Dataset):
    """Dataset COCO Captions avec features vision + texte pré-extraites."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        paired_fraction: float = 1.0,
    ):
        # Load pre-extracted vision features (ResNet-50)
        features_path = os.path.join(data_root, f"features_{split}.npy")
        ids_path = os.path.join(data_root, f"image_ids_{split}.json")

        self.features = np.load(features_path)  # (N_images, 2048)
        with open(ids_path) as f:
            image_ids = json.load(f)
        self.id_to_idx = {img_id: idx for idx, img_id in enumerate(image_ids)}

        # Load pre-extracted text features (E5-large-v2)
        text_features_path = os.path.join(data_root, f"text_features_{split}.npy")
        caption_ids_path = os.path.join(data_root, f"caption_ids_{split}.json")

        self.text_features = np.load(text_features_path)  # (N_captions, 1024)
        with open(caption_ids_path) as f:
            caption_samples = json.load(f)

        # Build (image_id, text_feature_idx) pairs
        # caption_samples is a list of {"image_id": ..., "caption": ...}
        # aligned with text_features by index
        self.samples = []
        for i, sample in enumerate(caption_samples):
            if sample["image_id"] in self.id_to_idx:
                self.samples.append((sample["image_id"], i))

        # Paired fraction for semi-supervised training
        self.paired_fraction = paired_fraction

        num_paired = sum(1 for img_id, _ in self.samples if self._is_paired(img_id))
        print(
            f"COCODataset({split}): {len(self.samples)} pairs, "
            f"{len(self.id_to_idx)} images, "
            f"paired={num_paired} ({100*num_paired/len(self.samples):.1f}%)"
        )

    def _is_paired(self, image_id: int) -> bool:
        """Déterministe : 20% des images sont paired (hash stable)."""
        if self.paired_fraction >= 1.0:
            return True
        # hash(int) is deterministic in Python for ints
        return (image_id % 5) == 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_id, text_idx = self.samples[idx]

        # Vision: pre-extracted ResNet-50 features
        feat_idx = self.id_to_idx[image_id]
        vision = torch.tensor(self.features[feat_idx], dtype=torch.float32)
        vision = F.normalize(vision, dim=-1)

        # Text: pre-extracted E5-large-v2 embeddings
        text = torch.tensor(self.text_features[text_idx], dtype=torch.float32)

        return {
            "vision": vision,
            "text": text,
            "image_id": torch.tensor(image_id, dtype=torch.long),
            "is_paired": torch.tensor(self._is_paired(image_id), dtype=torch.bool),
        }


class DummyMultimodalDataset(Dataset):
    """Dataset de test avec données aléatoires (pour debug)."""

    def __init__(
        self,
        num_samples: int = 10000,
        vision_dim: int = 2048,
        text_dim: int = 1024,
    ):
        self.num_samples = num_samples
        self.vision_dim = vision_dim
        self.text_dim = text_dim

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(idx)
        return {
            "vision": torch.rand(self.vision_dim, generator=gen),
            "text": torch.rand(self.text_dim, generator=gen),
            "image_id": torch.tensor(idx, dtype=torch.long),
            "is_paired": torch.tensor(True, dtype=torch.bool),
        }


def get_dataset(config: dict, split: str = "train") -> Dataset:
    """Factory : retourne le bon dataset selon la config."""
    name = config["data"]["dataset"]

    if name == "dummy":
        return DummyMultimodalDataset(
            vision_dim=config["vision"]["feature_dim"],
            text_dim=config["text"]["feature_dim"],
        )
    elif name == "coco":
        return COCODataset(
            data_root=config["data"]["data_root"],
            split=split,
            paired_fraction=config["training"].get("paired_fraction", 1.0),
        )
    else:
        raise ValueError(
            f"Dataset '{name}' pas implémenté. Utilise 'dummy' ou 'coco'."
        )
