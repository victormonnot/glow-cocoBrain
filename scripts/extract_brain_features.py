"""
extract_brain_features.py — Extrait les features fMRI (NSD) pour Subject 1
============================================================================
Usage: python scripts/extract_brain_features.py [--nsd_root data/nsd_raw]
                                                [--coco_root data/coco]
                                                [--output_root data/nsd]
                                                [--n_components 1024]

Pipeline:
  1. Charge le masque ROI nsdgeneral → indices voxels actifs
  2. Pour chaque session (1-40), charge les betas, extrait les voxels masqués
  3. Mappe chaque trial → cocoId via nsd_stim_info_merged.csv
  4. Moyenne les betas par image unique (~3 repetitions par image)
  5. Split train/val selon les image_ids COCO existants
  6. PCA sur train, applique à train + val
  7. Sauvegarde .npy + metadata

Produit:
  - data/nsd/brain_features_train.npy    (N_train, 1024) float32
  - data/nsd/brain_features_val.npy      (N_val, 1024) float32
  - data/nsd/brain_image_ids_train.json  [cocoId, ...]
  - data/nsd/brain_image_ids_val.json
  - data/nsd/pca_model.pkl               sklearn PCA fitted on train
  - data/nsd/brain_meta.json             metadata

Nécessite: pip install nibabel scikit-learn scipy
"""

import argparse
import json
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.io import loadmat


def load_roi_mask(nsd_root: str) -> np.ndarray:
    """Charge le masque ROI nsdgeneral et retourne les indices des voxels actifs."""
    import nibabel as nib

    mask_path = os.path.join(
        nsd_root, "nsddata/ppdata/subj01/func1pt8mm/roi/nsdgeneral.nii.gz"
    )
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()

    # nsdgeneral: voxels > 0 sont dans le ROI
    roi_indices = np.where(mask_data.flatten() > 0)[0]
    print(f"  ROI nsdgeneral: {len(roi_indices)} voxels (shape: {mask_data.shape})")
    return roi_indices, mask_data.shape


def load_trial_mapping(nsd_root: str) -> dict:
    """Charge le mapping trial → cocoId depuis nsd_stim_info_merged.csv.

    Returns:
        nsd_id_to_coco: dict mapping 0-indexed NSD image index → COCO image_id
    """
    stim_info_path = os.path.join(
        nsd_root, "nsddata/experiments/nsd/nsd_stim_info_merged.csv"
    )
    df = pd.read_csv(stim_info_path)

    # nsdId is 0-indexed, cocoId is the COCO image_id
    nsd_id_to_coco = dict(zip(df["nsdId"], df["cocoId"]))
    print(f"  Stimulus mapping: {len(nsd_id_to_coco)} NSD images → COCO IDs")
    return nsd_id_to_coco


def load_exp_design(nsd_root: str) -> np.ndarray:
    """Charge nsd_expdesign.mat pour obtenir l'ordre des stimuli par session.

    Returns:
        masterordering: (N_trials_total,) array of 0-indexed NSD image indices
        Each session has 750 trials (except possibly the last).
    """
    mat_path = os.path.join(
        nsd_root, "nsddata/experiments/nsd/nsd_expdesign.mat"
    )
    mat = loadmat(mat_path)

    # masterordering: (N, 1) array, 1-indexed → convert to 0-indexed
    masterordering = mat["masterordering"].flatten() - 1

    # masterordering has exactly 30000 entries (one per trial for one subject)
    # 40 sessions × 750 trials = 30000
    n_trials_subj1 = len(masterordering)

    print(f"  Master ordering: {len(masterordering)} trials")
    print(f"  NSD image indices: {masterordering.min()}-{masterordering.max()} (0-indexed)")

    return masterordering, n_trials_subj1


def extract_betas_all_sessions(
    nsd_root: str,
    roi_indices: np.ndarray,
    volume_shape: tuple,
    n_trials_subj1: int,
) -> np.ndarray:
    """Charge les betas de toutes les sessions et extrait les voxels ROI.

    Returns:
        all_betas: (n_trials_subj1, n_voxels) float32
    """
    import nibabel as nib

    betas_dir = os.path.join(
        nsd_root, "nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR"
    )

    n_voxels = len(roi_indices)
    all_betas = []
    trial_count = 0

    for session in range(1, 41):
        beta_path = os.path.join(betas_dir, f"betas_session{session:02d}.nii.gz")
        if not os.path.exists(beta_path):
            print(f"  [SKIP] Session {session} not found")
            continue

        print(f"  Loading session {session:02d}...", end=" ", flush=True)
        beta_img = nib.load(beta_path)
        beta_data = beta_img.get_fdata()  # (x, y, z, n_trials_session)

        n_trials_session = beta_data.shape[3]
        print(f"{n_trials_session} trials")

        # Reshape to (n_trials, n_voxels_total) then select ROI
        beta_flat = beta_data.reshape(-1, n_trials_session).T  # (n_trials, n_voxels_total)
        beta_roi = beta_flat[:, roi_indices]  # (n_trials, n_voxels_roi)

        all_betas.append(beta_roi.astype(np.float32))
        trial_count += n_trials_session

        if trial_count >= n_trials_subj1:
            break

    all_betas = np.concatenate(all_betas, axis=0)[:n_trials_subj1]
    print(f"  Total betas extracted: {all_betas.shape}")
    return all_betas


def average_by_image(
    all_betas: np.ndarray,
    masterordering: np.ndarray,
    nsd_id_to_coco: dict,
    n_trials: int,
) -> tuple:
    """Moyenne les betas par image unique (chaque image vue ~3 fois).

    Returns:
        coco_ids: list of COCO image IDs (sorted)
        averaged_betas: (n_unique_images, n_voxels) float32
    """
    # Group trials by NSD image index
    image_betas = defaultdict(list)
    for trial_idx in range(n_trials):
        nsd_id = masterordering[trial_idx]
        if nsd_id in nsd_id_to_coco:
            image_betas[nsd_id].append(all_betas[trial_idx])

    # Average and convert to COCO IDs
    coco_id_to_beta = {}
    for nsd_id, betas_list in image_betas.items():
        coco_id = nsd_id_to_coco[nsd_id]
        coco_id_to_beta[coco_id] = np.mean(betas_list, axis=0)

    # Sort by COCO ID for reproducibility
    coco_ids = sorted(coco_id_to_beta.keys())
    averaged_betas = np.stack([coco_id_to_beta[cid] for cid in coco_ids])

    n_reps = np.mean([len(v) for v in image_betas.values()])
    print(f"  {len(coco_ids)} unique images (avg {n_reps:.1f} reps/image)")
    print(f"  Averaged betas shape: {averaged_betas.shape}")

    return coco_ids, averaged_betas


def split_train_val(
    coco_ids: list,
    averaged_betas: np.ndarray,
    coco_root: str,
) -> tuple:
    """Split en train/val selon les image_ids COCO existants."""
    # Load existing COCO splits
    with open(os.path.join(coco_root, "image_ids_train.json")) as f:
        coco_train_ids = set(json.load(f))
    with open(os.path.join(coco_root, "image_ids_val.json")) as f:
        coco_val_ids = set(json.load(f))

    train_indices = []
    val_indices = []
    train_coco_ids = []
    val_coco_ids = []

    for i, cid in enumerate(coco_ids):
        if cid in coco_train_ids:
            train_indices.append(i)
            train_coco_ids.append(cid)
        elif cid in coco_val_ids:
            val_indices.append(i)
            val_coco_ids.append(cid)
        # else: NSD image not in COCO train or val — skip

    train_betas = averaged_betas[train_indices]
    val_betas = averaged_betas[val_indices]

    print(f"  Train: {len(train_coco_ids)} images, Val: {len(val_coco_ids)} images")
    print(f"  Skipped (not in COCO splits): {len(coco_ids) - len(train_coco_ids) - len(val_coco_ids)}")

    return train_coco_ids, train_betas, val_coco_ids, val_betas


def fit_pca(train_betas: np.ndarray, n_components: int) -> tuple:
    """Fit PCA sur les betas train, retourne le modèle et les betas transformés."""
    from sklearn.decomposition import PCA

    print(f"  Fitting PCA({n_components}) on {train_betas.shape}...")
    pca = PCA(n_components=n_components)
    train_reduced = pca.fit_transform(train_betas).astype(np.float32)

    explained = pca.explained_variance_ratio_.sum()
    print(f"  Explained variance: {explained:.4f} ({explained*100:.1f}%)")
    print(f"  Output shape: {train_reduced.shape}")

    return pca, train_reduced


def zscore_normalize(train: np.ndarray, val: np.ndarray) -> tuple:
    """Z-score normalize per dimension (fit on train, apply to both).

    Standard in brain decoding literature (MindEye, Brain-Diffuser).
    Returns normalized arrays + (mean, std) for inverse transform.
    """
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-8] = 1.0  # avoid division by zero for constant dims

    train_norm = ((train - mean) / std).astype(np.float32)
    val_norm = ((val - mean) / std).astype(np.float32)

    print(f"  Z-score normalization:")
    print(f"    Before — train mean: {train.mean():.2f}, std: {train.std():.2f}, norm: {np.linalg.norm(train, axis=1).mean():.2f}")
    print(f"    After  — train mean: {train_norm.mean():.4f}, std: {train_norm.std():.4f}, norm: {np.linalg.norm(train_norm, axis=1).mean():.2f}")

    return train_norm, val_norm, mean, std


def main():
    parser = argparse.ArgumentParser(
        description="Extract NSD brain features (Subject 1)"
    )
    parser.add_argument("--nsd_root", type=str, default="data/nsd_raw")
    parser.add_argument("--coco_root", type=str, default="data/coco")
    parser.add_argument("--output_root", type=str, default="data/nsd")
    parser.add_argument("--n_components", type=int, default=1024)
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    print("=== Extraction des brain features NSD ===")
    print()

    # 1. Load ROI mask
    print("--- ROI mask ---")
    roi_indices, volume_shape = load_roi_mask(args.nsd_root)

    # 2. Load trial → cocoId mapping
    print("\n--- Trial mapping ---")
    nsd_id_to_coco = load_trial_mapping(args.nsd_root)

    # 3. Load experiment design (trial ordering)
    print("\n--- Experiment design ---")
    masterordering, n_trials_subj1 = load_exp_design(args.nsd_root)

    # 4. Extract betas from all sessions
    print("\n--- Extracting betas (this may take a while) ---")
    all_betas = extract_betas_all_sessions(
        args.nsd_root, roi_indices, volume_shape, n_trials_subj1
    )

    # 5. Average by unique image
    print("\n--- Averaging by image ---")
    coco_ids, averaged_betas = average_by_image(
        all_betas, masterordering, nsd_id_to_coco, n_trials_subj1
    )

    # Free memory
    del all_betas

    # 6. Split train/val
    print("\n--- Train/Val split ---")
    train_ids, train_betas, val_ids, val_betas = split_train_val(
        coco_ids, averaged_betas, args.coco_root
    )

    del averaged_betas

    # 7. PCA
    print("\n--- PCA ---")
    pca, train_reduced = fit_pca(train_betas, args.n_components)
    val_reduced = pca.transform(val_betas).astype(np.float32)
    print(f"  Val reduced shape: {val_reduced.shape}")

    # 8. Z-score normalize (critical for GW training — match vision/text scale)
    print("\n--- Z-score normalization ---")
    train_reduced, val_reduced, zscore_mean, zscore_std = zscore_normalize(
        train_reduced, val_reduced
    )

    # 9. Save everything
    print("\n--- Saving ---")

    np.save(os.path.join(args.output_root, "brain_features_train.npy"), train_reduced)
    np.save(os.path.join(args.output_root, "brain_features_val.npy"), val_reduced)
    print(f"  brain_features_train.npy: {train_reduced.shape}")
    print(f"  brain_features_val.npy: {val_reduced.shape}")

    with open(os.path.join(args.output_root, "brain_image_ids_train.json"), "w") as f:
        json.dump(train_ids, f)
    with open(os.path.join(args.output_root, "brain_image_ids_val.json"), "w") as f:
        json.dump(val_ids, f)

    with open(os.path.join(args.output_root, "pca_model.pkl"), "wb") as f:
        pickle.dump(pca, f)

    # Save z-score stats for inverse transform (brain encoding → raw betas)
    np.save(os.path.join(args.output_root, "zscore_mean.npy"), zscore_mean)
    np.save(os.path.join(args.output_root, "zscore_std.npy"), zscore_std)

    meta = {
        "subject": 1,
        "roi": "nsdgeneral",
        "n_voxels_raw": len(roi_indices),
        "n_components": args.n_components,
        "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_unique_images_total": len(coco_ids),
        "zscore_normalized": True,
    }
    with open(os.path.join(args.output_root, "brain_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  brain_meta.json: {meta}")

    print("\nDone.")


if __name__ == "__main__":
    main()
