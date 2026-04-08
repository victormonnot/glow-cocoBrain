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
    all_w = {"vision": [], "text": []}
    all_z = {"vision": [], "text": []}
    all_x = {"vision": [], "text": []}

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

    # Centroid cosine similarity
    centroid_vis = all_w["vision"].mean(dim=0)
    centroid_text = all_w["text"].mean(dim=0)
    metrics["centroid_cos"] = F.cosine_similarity(
        centroid_vis.unsqueeze(0), centroid_text.unsqueeze(0)
    ).item()

    # Pairwise cosine: positives (matched pairs) and negatives
    w_vis_norm = F.normalize(all_w["vision"], dim=-1)
    w_text_norm = F.normalize(all_w["text"], dim=-1)

    # Positives: diagonal of similarity matrix
    cos_pos = (w_vis_norm * w_text_norm).sum(dim=-1)  # (N,)
    metrics["pairwise_cos_pos"] = cos_pos.mean().item()

    # Negatives: off-diagonal mean (sample for efficiency)
    N = w_vis_norm.size(0)
    sim_matrix = torch.matmul(w_vis_norm, w_text_norm.T)  # (N, N)
    # Mask out diagonal
    mask = ~torch.eye(N, device=device, dtype=torch.bool)
    metrics["pairwise_cos_neg"] = sim_matrix[mask].mean().item()

    # Gap between pos and neg (higher = better discrimination)
    metrics["cos_gap"] = metrics["pairwise_cos_pos"] - metrics["pairwise_cos_neg"]

    # ================================================================
    # LOSSES (decomposed)
    # ================================================================

    w_vis = all_w["vision"]
    w_text = all_w["text"]
    z_vis = all_z["vision"]
    z_text = all_z["text"]

    mod_vis = workspace.domain_modules["vision"]
    mod_text = workspace.domain_modules["text"]

    # Contrastive (Lcont)
    metrics["loss_contrastive"] = contrastive_fn(w_vis, w_text).item()

    # Demi-cycle (Ldcy) — on latents
    z_text_hat = mod_text.from_workspace(w_vis)
    metrics["loss_demi_cycle_v2t"] = F.mse_loss(z_text_hat, z_text).item()

    z_vis_hat = mod_vis.from_workspace(w_text)
    metrics["loss_demi_cycle_t2v"] = F.mse_loss(z_vis_hat, z_vis).item()

    # Translation (Ltr) — through decoder (observation space), MSE for both
    x_vis_hat = mod_vis.decode(z_vis_hat)
    metrics["loss_translation_t2v"] = F.mse_loss(x_vis_hat, all_x["vision"]).item()

    x_text_hat = mod_text.decode(z_text_hat)
    metrics["loss_translation_v2t"] = F.mse_loss(x_text_hat, all_x["text"]).item()

    # Cycle (Lcy)
    z_vis_cycle = compute_cycle(mod_vis, mod_text, z_vis)
    z_text_cycle = compute_cycle(mod_text, mod_vis, z_text)
    metrics["loss_cycle"] = (
        F.mse_loss(z_vis_cycle, z_vis).item()
        + F.mse_loss(z_text_cycle, z_text).item()
    ) / 2

    # Fusion (Lfusion) — stress test at alpha=0.5 with Tanh
    w_fused_half = torch.tanh(0.5 * w_vis + 0.5 * w_text)
    z_vis_from_fused = mod_vis.from_workspace(w_fused_half)
    z_text_from_fused = mod_text.from_workspace(w_fused_half)
    metrics["loss_fusion_vis"] = F.mse_loss(z_vis_from_fused, z_vis).item()
    metrics["loss_fusion_text"] = F.mse_loss(z_text_from_fused, z_text).item()
    metrics["loss_fusion_total"] = (metrics["loss_fusion_vis"] + metrics["loss_fusion_text"]) / 2

    # ================================================================
    # RETRIEVAL (on this val subset)
    # ================================================================

    # Image -> Text
    sim = torch.matmul(w_vis_norm, w_text_norm.T)  # already computed
    gt = torch.arange(N, device=device)

    _, topk_i2t = sim.topk(5, dim=1)
    metrics["retrieval_i2t_R@1"] = (topk_i2t[:, :1] == gt.unsqueeze(1)).any(dim=1).float().mean().item()
    metrics["retrieval_i2t_R@5"] = (topk_i2t[:, :5] == gt.unsqueeze(1)).any(dim=1).float().mean().item()

    # Text -> Image
    _, topk_t2i = sim.T.topk(5, dim=1)
    metrics["retrieval_t2i_R@1"] = (topk_t2i[:, :1] == gt.unsqueeze(1)).any(dim=1).float().mean().item()
    metrics["retrieval_t2i_R@5"] = (topk_t2i[:, :5] == gt.unsqueeze(1)).any(dim=1).float().mean().item()

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
