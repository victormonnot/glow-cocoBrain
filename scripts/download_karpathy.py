"""
download_karpathy.py — Télécharge le split Karpathy pour COCO
==============================================================
Télécharge dataset_coco.json (train/val/test splits de Karpathy)
et extrait les image IDs du test split (5000 images).

Sauvegarde:
  data/coco/karpathy_test_ids.json  — liste des 5000 image IDs du test split

Usage:
    python scripts/download_karpathy.py
"""

import json
import os
import urllib.request
import zipfile
import tempfile

DATA_ROOT = "data/coco"


def main():
    os.makedirs(DATA_ROOT, exist_ok=True)
    output_path = os.path.join(DATA_ROOT, "karpathy_test_ids.json")

    if os.path.exists(output_path):
        with open(output_path) as f:
            ids = json.load(f)
        print(f"Karpathy test IDs deja presents: {output_path} ({len(ids)} images)")
        return

    url = "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
    print(f"Telechargement: {url}")

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "caption_datasets.zip")
        urllib.request.urlretrieve(url, zip_path)
        print(f"Telecharge: {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extract("dataset_coco.json", tmpdir)

        with open(os.path.join(tmpdir, "dataset_coco.json")) as f:
            data = json.load(f)

    # Extraire les image IDs du test split
    test_ids = []
    for img in data["images"]:
        if img["split"] == "test":
            test_ids.append(img["cocoid"])

    test_ids.sort()
    print(f"Test split: {len(test_ids)} images")

    with open(output_path, "w") as f:
        json.dump(test_ids, f)
    print(f"Sauvegarde: {output_path}")


if __name__ == "__main__":
    main()
