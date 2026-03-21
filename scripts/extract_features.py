"""
extract_features.py — Extrait les features ResNet-50 pour toutes les images COCO
==================================================================================
Usage: python scripts/extract_features.py [--data_root data/coco] [--batch_size 128]

Produit pour chaque split:
  - data/coco/features_{split}.npy     (N, 2048) float32
  - data/coco/image_ids_{split}.json   mapping index → image_id
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


class ImageDataset(Dataset):
    """Charge des images depuis une liste de paths."""

    def __init__(self, image_dir: str, image_ids: list, transform):
        self.image_dir = image_dir
        self.image_ids = image_ids
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        path = os.path.join(self.image_dir, f"{img_id:012d}.jpg")
        img = Image.open(path).convert("RGB")
        return self.transform(img), img_id


def extract_split(model, split: str, data_root: str, batch_size: int, device: torch.device):
    """Extrait les features pour un split (train ou val)."""
    ann_path = os.path.join(data_root, "annotations", f"captions_{split}2017.json")
    image_dir = os.path.join(data_root, f"{split}2017")

    features_path = os.path.join(data_root, f"features_{split}.npy")
    ids_path = os.path.join(data_root, f"image_ids_{split}.json")

    if os.path.exists(features_path):
        print(f"  {features_path} already exists, skipping.")
        return

    with open(ann_path) as f:
        annotations = json.load(f)

    # Unique image IDs
    image_ids = sorted(set(img["id"] for img in annotations["images"]))
    print(f"  {split}: {len(image_ids)} images")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageDataset(image_dir, image_ids, transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    all_features = []
    all_ids = []

    with torch.no_grad():
        for i, (images, ids) in enumerate(loader):
            features = model(images.to(device))  # (B, 2048)
            all_features.append(features.cpu().numpy())
            all_ids.extend(ids.tolist())

            if (i + 1) % 50 == 0:
                print(f"    {(i + 1) * batch_size}/{len(dataset)}")

    all_features = np.concatenate(all_features, axis=0)  # (N, 2048)
    np.save(features_path, all_features)

    with open(ids_path, "w") as f:
        json.dump(all_ids, f)

    print(f"  Saved {features_path}: {all_features.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/coco")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ResNet-50 sans la couche fc (output = avgpool 2048-dim)
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet.fc = torch.nn.Identity()
    resnet = resnet.to(device)
    resnet.eval()

    print("=== Extraction des features ResNet-50 ===")
    for split in ["train", "val"]:
        extract_split(resnet, split, args.data_root, args.batch_size, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
