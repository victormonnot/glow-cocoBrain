"""
nsd_dataset.py — Dataset NSD 3-modalités pour Phase 3 (brain alignment)
========================================================================
Charge les features pré-extraites de 3 modalités :
  - vision : ResNet-50 features (2048) depuis data/coco/
  - text   : E5-large-v2 embeddings (1024) depuis data/coco/
  - brain  : fMRI betas PCA-réduits (1024) depuis data/nsd/

Seulement les ~10k images qui ont des données fMRI (NSD Subject 1).
Chaque image a ~5 captions COCO → les samples sont aplatis comme COCODataset.
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Dict


class NSDDataset(Dataset):
    """Dataset 3-modalités : vision + text + brain pour les images NSD."""

    def __init__(
        self,
        coco_root: str,
        nsd_root: str,
        split: str = "train",
        paired_fraction: float = 1.0,
    ):
        # --- Brain features (PCA-reduced fMRI betas) ---
        brain_features_path = os.path.join(nsd_root, f"brain_features_{split}.npy")
        brain_ids_path = os.path.join(nsd_root, f"brain_image_ids_{split}.json")

        self.brain_features = np.load(brain_features_path)  # (N_brain, 1024)
        with open(brain_ids_path) as f:
            brain_image_ids = json.load(f)
        self.brain_id_to_idx = {img_id: i for i, img_id in enumerate(brain_image_ids)}

        # --- Vision features (ResNet-50) ---
        vision_features_path = os.path.join(coco_root, f"features_{split}.npy")
        vision_ids_path = os.path.join(coco_root, f"image_ids_{split}.json")

        self.vision_features = np.load(vision_features_path)  # (N_images, 2048)
        with open(vision_ids_path) as f:
            coco_image_ids = json.load(f)
        self.vision_id_to_idx = {img_id: i for i, img_id in enumerate(coco_image_ids)}

        # --- Text features (E5-large-v2) ---
        text_features_path = os.path.join(coco_root, f"text_features_{split}.npy")
        caption_ids_path = os.path.join(coco_root, f"caption_ids_{split}.json")

        self.text_features = np.load(text_features_path)  # (N_captions, 1024)
        with open(caption_ids_path) as f:
            caption_samples = json.load(f)

        # Group captions by image_id
        caption_by_image = {}
        for i, sample in enumerate(caption_samples):
            img_id = sample["image_id"]
            if img_id not in caption_by_image:
                caption_by_image[img_id] = []
            caption_by_image[img_id].append(i)

        # Build samples: only images that exist in brain AND vision AND text
        self.samples = []
        for img_id in brain_image_ids:
            if img_id in self.vision_id_to_idx and img_id in caption_by_image:
                for text_idx in caption_by_image[img_id]:
                    self.samples.append((img_id, text_idx))

        self.paired_fraction = paired_fraction

        # Stats
        n_images = len(set(img_id for img_id, _ in self.samples))
        n_paired = sum(1 for img_id, _ in self.samples if self._is_paired(img_id))
        print(
            f"NSDDataset({split}): {len(self.samples)} pairs, "
            f"{n_images} images (brain+vision+text), "
            f"paired={n_paired} ({100*n_paired/max(len(self.samples),1):.1f}%)"
        )

    def _is_paired(self, image_id: int) -> bool:
        """Déterministe : même logique que COCODataset."""
        if self.paired_fraction >= 1.0:
            return True
        return (image_id % 5) == 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_id, text_idx = self.samples[idx]

        # Vision: pre-extracted ResNet-50 features
        vis_idx = self.vision_id_to_idx[image_id]
        vision = torch.tensor(self.vision_features[vis_idx], dtype=torch.float32)
        vision = F.normalize(vision, dim=-1)

        # Text: pre-extracted E5-large-v2 embeddings
        text = torch.tensor(self.text_features[text_idx], dtype=torch.float32)

        # Brain: pre-extracted PCA-reduced fMRI betas
        brain_idx = self.brain_id_to_idx[image_id]
        brain = torch.tensor(self.brain_features[brain_idx], dtype=torch.float32)

        return {
            "vision": vision,
            "text": text,
            "brain": brain,
            "image_id": torch.tensor(image_id, dtype=torch.long),
            "is_paired": torch.tensor(self._is_paired(image_id), dtype=torch.bool),
        }
