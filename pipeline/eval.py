"""
eval.py — Évaluation du Global Workspace
==========================================
Métriques :
  1. Cross-modal retrieval (R@1, R@5, R@10)
  2. Translation MSE (vision et texte — les deux sont MSE maintenant)
  3. Stats workspace (normes, distances centroïdes)
  4. Karpathy 1K retrieval (standard benchmark, 5-fold sur 5K test)
"""

import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from typing import Dict, List

from models.workspace import GlobalWorkspace


@torch.no_grad()
def compute_retrieval_metrics(
    workspace: GlobalWorkspace,
    dataloader: DataLoader,
    source_domain: str,
    target_domain: str,
    device: torch.device,
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """Cross-modal retrieval : Recall@K."""
    workspace.eval()

    all_w_source = []
    all_w_target = []

    for batch in dataloader:
        x_source = batch[source_domain].to(device)
        x_target = batch[target_domain].to(device)

        w_source = workspace.domain_modules[source_domain].encode_to_workspace(x_source)
        w_target = workspace.domain_modules[target_domain].encode_to_workspace(x_target)

        all_w_source.append(F.normalize(w_source, dim=-1))
        all_w_target.append(F.normalize(w_target, dim=-1))

    all_w_source = torch.cat(all_w_source, dim=0)
    all_w_target = torch.cat(all_w_target, dim=0)

    # Matrice de similarité (N, N)
    sim_matrix = torch.matmul(all_w_source, all_w_target.T)

    N = sim_matrix.size(0)
    gt_indices = torch.arange(N, device=device)

    _, topk_indices = sim_matrix.topk(max(k_values), dim=1)

    results = {}
    for k in k_values:
        correct = (topk_indices[:, :k] == gt_indices.unsqueeze(1)).any(dim=1)
        results[f"R@{k}"] = correct.float().mean().item()

    return results


@torch.no_grad()
def compute_translation_mse(
    workspace: GlobalWorkspace,
    dataloader: DataLoader,
    source_domain: str,
    target_domain: str,
    device: torch.device,
) -> float:
    """MSE de traduction cross-modale (fonctionne pour vision et texte)."""
    workspace.eval()
    total_mse = 0.0
    count = 0

    for batch in dataloader:
        x_source = batch[source_domain].to(device)
        x_target = batch[target_domain].to(device)

        x_target_hat = workspace.translate(source_domain, target_domain, x_source)
        total_mse += F.mse_loss(x_target_hat, x_target, reduction="sum").item()
        count += x_target.size(0)

    return total_mse / count


@torch.no_grad()
def compute_workspace_stats(
    workspace: GlobalWorkspace,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Stats de l'espace workspace pour debug."""
    workspace.eval()

    domain_reprs = {name: [] for name in workspace.domain_modules}

    for batch in dataloader:
        reprs = workspace.encode_domains(
            {name: batch[name].to(device) for name in batch if name in workspace.domain_modules}
        )
        for name, w in reprs.items():
            domain_reprs[name].append(w)

    stats = {}
    means = {}

    for name, repr_list in domain_reprs.items():
        all_w = torch.cat(repr_list, dim=0)
        stats[f"{name}_mean_norm"] = all_w.norm(dim=-1).mean().item()
        stats[f"{name}_std"] = all_w.std().item()
        means[name] = all_w.mean(dim=0)

    # Distance entre les centroïdes des domaines
    names = list(means.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            dist = (means[names[i]] - means[names[j]]).norm().item()
            cos = F.cosine_similarity(
                means[names[i]].unsqueeze(0), means[names[j]].unsqueeze(0)
            ).item()
            stats[f"centroid_dist_{names[i]}_{names[j]}"] = dist
            stats[f"centroid_cos_{names[i]}_{names[j]}"] = cos

    return stats


@torch.no_grad()
def compute_karpathy_retrieval(
    workspace: GlobalWorkspace,
    val_dataset,
    config: dict,
    device: torch.device,
    n_folds: int = 5,
) -> Dict[str, float]:
    """Karpathy 1K retrieval: 5-fold sur les 5K images test, moyenné.

    Le protocole standard:
      - 5000 images test Karpathy, chaque image a ~5 captions
      - On split en 5 folds de 1000 images
      - Pour chaque fold: retrieval image→text et text→image sur les 1K images
      - On moyenne les R@1, R@5, R@10 sur les 5 folds
    """
    data_root = config["data"]["data_root"]
    karpathy_path = os.path.join(data_root, "karpathy_test_ids.json")

    if not os.path.exists(karpathy_path):
        print("  [Karpathy] karpathy_test_ids.json non trouvé, skip.")
        print("  Lancer: python scripts/download_karpathy.py")
        return {}

    with open(karpathy_path) as f:
        test_image_ids = set(json.load(f))

    # Trouver les indices du val_dataset qui correspondent aux images Karpathy test
    karpathy_indices = []
    for idx in range(len(val_dataset)):
        image_id, _ = val_dataset.samples[idx]
        if image_id in test_image_ids:
            karpathy_indices.append(idx)

    if len(karpathy_indices) == 0:
        print("  [Karpathy] Aucune image test trouvée dans le val set.")
        return {}

    print(f"  [Karpathy] {len(karpathy_indices)} paires (image, caption) trouvées")

    # Grouper par image_id pour le retrieval image-level
    image_to_caption_indices = {}
    for idx in karpathy_indices:
        image_id, _ = val_dataset.samples[idx]
        if image_id not in image_to_caption_indices:
            image_to_caption_indices[image_id] = []
        image_to_caption_indices[image_id].append(idx)

    unique_image_ids = sorted(image_to_caption_indices.keys())
    n_images = len(unique_image_ids)
    print(f"  [Karpathy] {n_images} images uniques")

    # Encode toutes les représentations workspace
    workspace.eval()
    batch_size = config["training"]["batch_size"]

    # Pour chaque image: un seul vecteur workspace (première caption)
    # Pour chaque caption: un vecteur workspace
    all_w_images = []  # (n_images, D)
    all_w_captions = []  # (n_captions_total, D)
    caption_to_image_idx = []  # mapping caption -> image index

    for img_idx, image_id in enumerate(unique_image_ids):
        caption_indices = image_to_caption_indices[image_id]

        # Vision: même features pour toutes les captions de cette image
        sample = val_dataset[caption_indices[0]]
        x_vis = sample["vision"].unsqueeze(0).to(device)
        w_vis = workspace.domain_modules["vision"].encode_to_workspace(x_vis)
        all_w_images.append(F.normalize(w_vis, dim=-1))

        # Text: chaque caption a son propre embedding
        for cap_idx in caption_indices:
            sample = val_dataset[cap_idx]
            x_txt = sample["text"].unsqueeze(0).to(device)
            w_txt = workspace.domain_modules["text"].encode_to_workspace(x_txt)
            all_w_captions.append(F.normalize(w_txt, dim=-1))
            caption_to_image_idx.append(img_idx)

    all_w_images = torch.cat(all_w_images, dim=0)  # (n_images, D)
    all_w_captions = torch.cat(all_w_captions, dim=0)  # (n_captions, D)
    caption_to_image_idx = torch.tensor(caption_to_image_idx, device=device)

    # 5-fold evaluation sur les images
    fold_size = n_images // n_folds
    k_values = [1, 5, 10]
    all_results = {f"i2t_R@{k}": [] for k in k_values}
    all_results.update({f"t2i_R@{k}": [] for k in k_values})

    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size
        fold_img_indices = list(range(start, end))

        # Masque des captions appartenant à ce fold
        fold_cap_mask = torch.zeros(len(caption_to_image_idx), dtype=torch.bool, device=device)
        for img_idx in fold_img_indices:
            fold_cap_mask |= (caption_to_image_idx == img_idx)

        w_img_fold = all_w_images[fold_img_indices]  # (fold_size, D)
        w_cap_fold = all_w_captions[fold_cap_mask]  # (n_caps_fold, D)
        cap_to_img_fold = caption_to_image_idx[fold_cap_mask]

        # Remap image indices to 0..fold_size-1
        img_remap = {old: new for new, old in enumerate(fold_img_indices)}
        cap_to_img_local = torch.tensor(
            [img_remap[c.item()] for c in cap_to_img_fold], device=device
        )

        # Image → Text retrieval
        sim_i2t = torch.matmul(w_img_fold, w_cap_fold.T)  # (fold_size, n_caps)
        for k in k_values:
            _, topk = sim_i2t.topk(k, dim=1)
            # Pour chaque image, check si au moins une de ses captions est dans le top-k
            correct = 0
            for i in range(len(fold_img_indices)):
                retrieved_img_ids = cap_to_img_local[topk[i]]
                if (retrieved_img_ids == i).any():
                    correct += 1
            all_results[f"i2t_R@{k}"].append(correct / len(fold_img_indices))

        # Text → Image retrieval
        sim_t2i = torch.matmul(w_cap_fold, w_img_fold.T)  # (n_caps, fold_size)
        for k in k_values:
            _, topk = sim_t2i.topk(k, dim=1)
            correct = 0
            for i in range(len(w_cap_fold)):
                gt_img = cap_to_img_local[i]
                if (topk[i] == gt_img).any():
                    correct += 1
            all_results[f"t2i_R@{k}"].append(correct / len(w_cap_fold))

    # Moyenne sur les folds
    results = {}
    for key, fold_vals in all_results.items():
        results[f"karpathy_{key}"] = sum(fold_vals) / len(fold_vals)

    return results


def evaluate(
    workspace: GlobalWorkspace,
    eval_loader: DataLoader,
    config: dict,
    device: torch.device,
    val_dataset=None,
) -> Dict[str, float]:
    """Évaluation complète."""
    results = {}
    domain_pairs = [tuple(p) for p in config["training"]["domain_pairs"]]

    print("=" * 60)
    print("ÉVALUATION")
    print("=" * 60)

    for source, target in domain_pairs:
        # Retrieval
        retrieval = compute_retrieval_metrics(
            workspace, eval_loader, source, target, device
        )
        for k, v in retrieval.items():
            key = f"retrieval_{source}→{target}_{k}"
            results[key] = v
            print(f"  {key}: {v:.4f}")

        # Translation MSE (both directions, both modalities use MSE now)
        mse = compute_translation_mse(
            workspace, eval_loader, source, target, device
        )
        key = f"translation_mse_{source}→{target}"
        results[key] = mse
        print(f"  {key}: {mse:.6f}")

    # Workspace stats
    stats = compute_workspace_stats(workspace, eval_loader, device)
    results.update(stats)
    print()
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    # Karpathy 1K retrieval (5-fold sur 5K test)
    if val_dataset is not None:
        print()
        print("--- Karpathy 1K retrieval (5-fold) ---")
        karpathy = compute_karpathy_retrieval(workspace, val_dataset, config, device)
        results.update(karpathy)
        for k, v in karpathy.items():
            print(f"  {k}: {v:.4f}")

    return results
