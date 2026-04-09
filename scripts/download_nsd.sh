#!/bin/bash
# download_nsd.sh — Télécharge NSD Subject 1 (betas + metadata) depuis AWS S3
# Usage: bash scripts/download_nsd.sh
#
# Télécharge ~20 GB :
#   - 40 sessions de betas fMRI préprocessés (betas_fithrf_GLMdenoise_RR)
#   - ROI mask nsdgeneral
#   - Metadata : stim_info (mapping trial→cocoId), expdesign (trial ordering)

set -e

DATA_DIR="data/nsd_raw"
BASE_URL="https://natural-scenes-dataset.s3.amazonaws.com"

mkdir -p "$DATA_DIR"

download_file() {
    local remote_path="$1"
    local local_path="$DATA_DIR/$remote_path"
    local dir=$(dirname "$local_path")

    mkdir -p "$dir"

    # Skip if file exists AND is non-empty
    if [ -f "$local_path" ] && [ -s "$local_path" ]; then
        echo "  [SKIP] $remote_path (already exists)"
        return
    fi

    # Remove empty/corrupt files
    rm -f "$local_path"

    echo "  [DL] $remote_path ..."
    wget --continue --tries=5 --timeout=60 --waitretry=10 \
         -q --show-progress -O "$local_path" "$BASE_URL/$remote_path"

    # Verify download succeeded (non-empty file)
    if [ ! -s "$local_path" ]; then
        echo "  [ERROR] Download failed for $remote_path (empty file)"
        rm -f "$local_path"
        exit 1
    fi
}

echo "=== Téléchargement NSD Subject 1 ==="
echo ""

# --- Metadata ---
echo "--- Metadata ---"
download_file "nsddata/experiments/nsd/nsd_stim_info_merged.csv"
download_file "nsddata/experiments/nsd/nsd_expdesign.mat"

# --- ROI mask ---
echo ""
echo "--- ROI mask (nsdgeneral) ---"
download_file "nsddata/ppdata/subj01/func1pt8mm/roi/nsdgeneral.nii.gz"

# --- Betas (40 sessions, ~500MB each) ---
# NOTE: betas are under nsddata_betas/ (separate from nsddata/)
echo ""
echo "--- Betas Subject 1 (40 sessions, ~20 GB total) ---"
for i in $(seq -w 1 40); do
    download_file "nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session${i}.nii.gz"
done

# --- Vérification ---
echo ""
echo "=== Vérification ==="

check_file() {
    if [ -f "$DATA_DIR/$1" ]; then
        echo "  [OK] $1"
    else
        echo "  [MISSING] $1"
        exit 1
    fi
}

check_file "nsddata/experiments/nsd/nsd_stim_info_merged.csv"
check_file "nsddata/experiments/nsd/nsd_expdesign.mat"
check_file "nsddata/ppdata/subj01/func1pt8mm/roi/nsdgeneral.nii.gz"
check_file "nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session01.nii.gz"
check_file "nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session40.nii.gz"

echo ""
echo "NSD Subject 1 prêt dans $DATA_DIR"
echo "Prochaine étape : python scripts/extract_brain_features.py"
