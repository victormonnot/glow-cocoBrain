#!/bin/bash
# download_coco.sh — Télécharge COCO 2017 (train, val, annotations)
# Usage: bash scripts/download_coco.sh

set -e

DATA_DIR="data/coco"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=== Téléchargement COCO 2017 ==="

download_and_extract() {
    local name="$1"
    local url="$2"
    local zip="${name}.zip"

    if [ -d "$name" ] || [ -d "annotations" -a "$name" = "annotations_trainval2017" ]; then
        echo "$name already exists, skipping."
        return
    fi

    if [ ! -f "$zip" ]; then
        echo "Downloading $name..."
        wget -q --show-progress "$url"
    else
        echo "$zip already downloaded, skipping download."
    fi

    echo "Extracting $zip..."
    unzip -q "$zip"
    rm "$zip"
    echo "$name OK"
}

# Train images (~18 GB)
download_and_extract "train2017" "http://images.cocodataset.org/zips/train2017.zip"

# Val images (~1 GB)
download_and_extract "val2017" "http://images.cocodataset.org/zips/val2017.zip"

# Annotations (~241 MB)
download_and_extract "annotations_trainval2017" "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# Vérification
echo ""
echo "=== Vérification ==="
for f in train2017 val2017 annotations/captions_train2017.json annotations/captions_val2017.json; do
    if [ -e "$f" ]; then
        echo "  [OK] $f"
    else
        echo "  [MISSING] $f"
        exit 1
    fi
done

echo ""
echo "COCO 2017 prêt dans $DATA_DIR"
