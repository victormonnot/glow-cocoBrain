"""
diagnostics.py — Monitoring anti-degenerescence
=================================================
Calcule a chaque epoch sur un batch val fixe :

  Geometrie workspace :
    - centroid_cos(w_vis, w_text)
    - mean_norm par domaine
    - std_per_dim par domaine (collapse = std -> 0)
    - effective_rank par domaine (SVD)
    - pairwise_cosine_pos (paires matchees)
    - pairwise_cosine_neg (paires non-matchees)

  Losses decomposees :
    - contrastive, translation_v2t, translation_t2v, cycle

  Retrieval :
    - R@1, R@5 image->text et text->image sur subset val fixe
"""

import csv
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from models.workspace import GlobalWorkspace
from losses.contrastive import ContrastiveLoss
from losses.cycle import CycleConsistencyLoss, compute_cycle


def build_fixed_val_subset(val_dataset, num_samples: int = 1000, seed: int = 42):
    """Construit un subset val fixe pour les diagnostics."""
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(val_dataset), generator=gen)[:num_samples].tolist()
    return Subset(val_dataset, indices)


def _effective_rank(W: torch.Tensor) -> float:
    """Effective rank via Shannon entropy of normalized singular values.

    Roy & Bhattacharya 2007. Plus c'est haut, plus les dimensions
    sont utilisees uniformement. Collapse = rank -> 1.
    """
    # W: (N, D) — N samples, D dims
    s = torch.linalg.svdvals(W)
    # Normalize to probability distribution
    p = s / s.sum()
    p = p[p > 1e-10]  # avoid log(0)
    entropy = -(p * p.log()).sum()
    return entropy.exp().item()


@torch.no_grad()
def compute_diagnostics(
    workspace: GlobalWorkspace,
    val_loader: DataLoader,
    device: torch.device,
    contrastive_fn: Optional[ContrastiveLoss] = None,
) -> Dict[str, float]:
    """Calcule toutes les metriques de diagnostic sur le val set."""
    workspace.eval()

    if contrastive_fn is None:
        contrastive_fn = ContrastiveLoss(temperature=0.07)

    # Collect all workspace representations, latents, and raw inputs
    domain_names = list(workspace.domain_modules.keys())
    all_w = {name: [] for name in domain_names}
    all_z = {name: [] for name in domain_names}
    all_x = {name: [] for name in domain_names}

    for batch in val_loader:
        inputs = {
            name: batch[name].to(device)
            for name in batch
            if name in workspace.domain_modules
        }

        for name in inputs:
            mod = workspace.domain_modules[name]
            z = mod.encode(inputs[name])
            w = mod.to_workspace(z)
            all_x[name].append(inputs[name])
            all_z[name].append(z)
            all_w[name].append(w)

    all_x = {k: torch.cat(v, dim=0) for k, v in all_x.items()}
    all_w = {k: torch.cat(v, dim=0) for k, v in all_w.items()}
    all_z = {k: torch.cat(v, dim=0) for k, v in all_z.items()}

    metrics = {}

    # ================================================================
    # GEOMETRY
    # ================================================================

    # Mean norm per domain
    for name, w in all_w.items():
        metrics[f"norm_{name}"] = w.norm(dim=-1).mean().item()

    # Std per dim (mean across dims) — collapse detector
    for name, w in all_w.items():
        metrics[f"std_per_dim_{name}"] = w.std(dim=0).mean().item()

    # Effective rank per domain
    for name, w in all_w.items():
        # Center before SVD
        w_centered = w - w.mean(dim=0, keepdim=True)
        metrics[f"effective_rank_{name}"] = _effective_rank(w_centered)

    # Centroid cosine similarity — for all domain pairs
    centroids = {name: all_w[name].mean(dim=0) for name in domain_names}
    for i in range(len(domain_names)):
        for j in range(i + 1, len(domain_names)):
            n_i, n_j = domain_names[i], domain_names[j]
            cos = F.cosine_similarity(
                centroids[n_i].unsqueeze(0), centroids[n_j].unsqueeze(0)
            ).item()
            metrics[f"centroid_cos_{n_i}_{n_j}"] = cos
    # Backward compat: "centroid_cos" = first pair
    if len(domain_names) >= 2:
        metrics["centroid_cos"] = metrics[f"centroid_cos_{domain_names[0]}_{domain_names[1]}"]

    # Pairwise cosine: positives and negatives — for first two domains
    # (same-index pairs are positives, off-diagonal are negatives)
    d0, d1 = domain_names[0], domain_names[1]
    w_d0_norm = F.normalize(all_w[d0], dim=-1)
    w_d1_norm = F.normalize(all_w[d1], dim=-1)

    cos_pos = (w_d0_norm * w_d1_norm).sum(dim=-1)
    metrics["pairwise_cos_pos"] = cos_pos.mean().item()

    N = w_d0_norm.size(0)
    sim_matrix = torch.matmul(w_d0_norm, w_d1_norm.T)
    mask = ~torch.eye(N, device=device, dtype=torch.bool)
    metrics["pairwise_cos_neg"] = sim_matrix[mask].mean().item()

    metrics["cos_gap"] = metrics["pairwise_cos_pos"] - metrics["pairwise_cos_neg"]

    # ================================================================
    # LOSSES (decomposed)
    # ================================================================

    # Compute losses for all domain pairs
    for i in range(len(domain_names)):
        for j in range(i + 1, len(domain_names)):
            n_a, n_b = domain_names[i], domain_names[j]
            w_a, w_b = all_w[n_a], all_w[n_b]
            z_a, z_b = all_z[n_a], all_z[n_b]
            mod_a = workspace.domain_modules[n_a]
            mod_b = workspace.domain_modules[n_b]

            # Contrastive (Lcont)
            metrics[f"loss_contrastive_{n_a}_{n_b}"] = contrastive_fn(w_a, w_b).item()

            # Demi-cycle (Ldcy)
            z_b_hat = mod_b.from_workspace(w_a)
            metrics[f"loss_demi_cycle_{n_a}2{n_b}"] = F.mse_loss(z_b_hat, z_b).item()
            z_a_hat = mod_a.from_workspace(w_b)
            metrics[f"loss_demi_cycle_{n_b}2{n_a}"] = F.mse_loss(z_a_hat, z_a).item()

            # Translation (Ltr) — through decoder
            x_a_hat = mod_a.decode(z_a_hat)
            metrics[f"loss_translation_{n_b}2{n_a}"] = F.mse_loss(x_a_hat, all_x[n_a]).item()
            x_b_hat = mod_b.decode(z_b_hat)
            metrics[f"loss_translation_{n_a}2{n_b}"] = F.mse_loss(x_b_hat, all_x[n_b]).item()

            # Cycle (Lcy)
            z_a_cycle = compute_cycle(mod_a, mod_b, z_a)
            z_b_cycle = compute_cycle(mod_b, mod_a, z_b)
            metrics[f"loss_cycle_{n_a}_{n_b}"] = (
                F.mse_loss(z_a_cycle, z_a).item()
                + F.mse_loss(z_b_cycle, z_b).item()
            ) / 2

            # Fusion (Lfusion) — stress test at alpha=0.5 with Tanh
            w_fused_half = torch.tanh(0.5 * w_a + 0.5 * w_b)
            z_a_from_fused = mod_a.from_workspace(w_fused_half)
            z_b_from_fused = mod_b.from_workspace(w_fused_half)
            metrics[f"loss_fusion_{n_a}"] = F.mse_loss(z_a_from_fused, z_a).item()
            metrics[f"loss_fusion_{n_b}"] = F.mse_loss(z_b_from_fused, z_b).item()
            metrics[f"loss_fusion_{n_a}_{n_b}"] = (
                metrics[f"loss_fusion_{n_a}"] + metrics[f"loss_fusion_{n_b}"]
            ) / 2

    # Backward compat aliases for 2-domain case (vision+text)
    if "vision" in domain_names and "text" in domain_names:
        metrics.setdefault("loss_contrastive", metrics.get("loss_contrastive_vision_text", 0.0))
        metrics.setdefault("loss_demi_cycle_v2t", metrics.get("loss_demi_cycle_vision2text", 0.0))
        metrics.setdefault("loss_demi_cycle_t2v", metrics.get("loss_demi_cycle_text2vision", 0.0))
        metrics.setdefault("loss_translation_v2t", metrics.get("loss_translation_vision2text", 0.0))
        metrics.setdefault("loss_translation_t2v", metrics.get("loss_translation_text2vision", 0.0))
        metrics.setdefault("loss_cycle", metrics.get("loss_cycle_vision_text", 0.0))
        metrics.setdefault("loss_fusion_vis", metrics.get("loss_fusion_vision", 0.0))
        metrics.setdefault("loss_fusion_text", metrics.get("loss_fusion_text", 0.0))
        metrics.setdefault("loss_fusion_total", metrics.get("loss_fusion_vision_text", 0.0))

    # ================================================================
    # RETRIEVAL (on this val subset)
    # ================================================================

    # Retrieval for all domain pairs
    gt = torch.arange(N, device=device)
    for i in range(len(domain_names)):
        for j in range(i + 1, len(domain_names)):
            n_a, n_b = domain_names[i], domain_names[j]
            w_a_n = F.normalize(all_w[n_a], dim=-1)
            w_b_n = F.normalize(all_w[n_b], dim=-1)
            sim = torch.matmul(w_a_n, w_b_n.T)

            # a -> b
            _, topk_a2b = sim.topk(5, dim=1)
            metrics[f"retrieval_{n_a}2{n_b}_R@1"] = (topk_a2b[:, :1] == gt.unsqueeze(1)).any(dim=1).float().mean().item()
            metrics[f"retrieval_{n_a}2{n_b}_R@5"] = (topk_a2b[:, :5] == gt.unsqueeze(1)).any(dim=1).float().mean().item()

            # b -> a
            _, topk_b2a = sim.T.topk(5, dim=1)
            metrics[f"retrieval_{n_b}2{n_a}_R@1"] = (topk_b2a[:, :1] == gt.unsqueeze(1)).any(dim=1).float().mean().item()
            metrics[f"retrieval_{n_b}2{n_a}_R@5"] = (topk_b2a[:, :5] == gt.unsqueeze(1)).any(dim=1).float().mean().item()

    # Backward compat aliases
    if "vision" in domain_names and "text" in domain_names:
        metrics.setdefault("retrieval_i2t_R@1", metrics.get("retrieval_vision2text_R@1", 0.0))
        metrics.setdefault("retrieval_i2t_R@5", metrics.get("retrieval_vision2text_R@5", 0.0))
        metrics.setdefault("retrieval_t2i_R@1", metrics.get("retrieval_text2vision_R@1", 0.0))
        metrics.setdefault("retrieval_t2i_R@5", metrics.get("retrieval_text2vision_R@5", 0.0))

    workspace.train()
    return metrics


class DiagnosticsLogger:
    """Log les diagnostics en CSV + wandb a chaque epoch."""

    def __init__(self, run_dir: str, use_wandb: bool = False, filename: str = "diagnostics.csv"):
        self.path = os.path.join(run_dir, filename)
        self.fieldnames = None
        self.use_wandb = use_wandb

    def log(self, epoch: int, metrics: Dict[str, float]):
        row = {"epoch": epoch, **metrics}

        # CSV (toujours)
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in row.items()})

        # wandb
        if self.use_wandb:
            import wandb
            wandb.log({"val/" + k: v for k, v in metrics.items()}, step=epoch)
