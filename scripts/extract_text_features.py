"""
extract_text_features.py — Extrait les embeddings E5-large-v2 pour toutes les captions COCO
=============================================================================================
Usage: python scripts/extract_text_features.py [--data_root data/coco] [--batch_size 256]

Produit pour chaque split:
  - data/coco/text_features_{split}.npy   (N_captions, 1024) float32
  - data/coco/caption_ids_{split}.json    mapping index → (image_id, caption)

Nécessite: pip install sentence-transformers
"""

import argparse
import json
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def extract_split(model, split: str, data_root: str, batch_size: int):
    """Extrait les embeddings E5 pour un split (train ou val)."""
    ann_path = os.path.join(data_root, "annotations", f"captions_{split}2017.json")

    features_path = os.path.join(data_root, f"text_features_{split}.npy")
    ids_path = os.path.join(data_root, f"caption_ids_{split}.json")

    if os.path.exists(features_path):
        print(f"  {features_path} already exists, skipping.")
        return

    with open(ann_path) as f:
        annotations = json.load(f)

    # Build ordered list of (image_id, caption) — same order as COCODataset
    # Load image_ids to filter only images with extracted vision features
    image_ids_path = os.path.join(data_root, f"image_ids_{split}.json")
    with open(image_ids_path) as f:
        valid_image_ids = set(json.load(f))

    samples = []
    captions = []
    for ann in annotations["annotations"]:
        if ann["image_id"] in valid_image_ids:
            samples.append({"image_id": ann["image_id"], "caption": ann["caption"]})
            # E5 expects "query: " prefix for encoding
            captions.append("query: " + ann["caption"])

    print(f"  {split}: {len(captions)} captions")

    # Encode in batches
    all_embeddings = model.encode(
        captions,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=False,  # Keep raw embeddings
    )

    all_embeddings = np.array(all_embeddings, dtype=np.float32)
    np.save(features_path, all_embeddings)

    with open(ids_path, "w") as f:
        json.dump(samples, f)

    print(f"  Saved {features_path}: {all_embeddings.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/coco")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--model_name", type=str, default="intfloat/e5-large-v2",
        help="Sentence-transformer model name",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model_name, device=device)
    print(f"  Embedding dim: {model.get_sentence_embedding_dimension()}")
    print(f"  Device: {device}")

    print("=== Extraction des embeddings texte ===")
    for split in ["train", "val"]:
        extract_split(model, split, args.data_root, args.batch_size)

    print("\nDone.")


if __name__ == "__main__":
    main()
