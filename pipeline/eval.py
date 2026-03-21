"""
eval.py — Évaluation du Global Workspace
==========================================
Métriques :
  1. Cross-modal retrieval (R@1, R@5, R@10)
  2. Translation MSE (pour vision)
  3. Traduction qualitative image → texte
  4. Stats workspace (normes, distances centroïdes)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional

from models.workspace import GlobalWorkspace
from data.tokenizer import SimpleTokenizer


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
    """MSE de traduction cross-modale."""
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
def show_translation_examples(
    workspace: GlobalWorkspace,
    dataloader: DataLoader,
    tokenizer: SimpleTokenizer,
    device: torch.device,
    num_examples: int = 10,
):
    """Affiche des exemples de traduction image → texte."""
    workspace.eval()
    batch = next(iter(dataloader))
    x_vision = batch["vision"][:num_examples].to(device)
    x_text = batch["text"][:num_examples]

    # Translate vision → text
    text_logits = workspace.translate("vision", "text", x_vision)
    predicted_ids = text_logits.argmax(dim=-1)

    print("  --- Exemples image → texte ---")
    for i in range(num_examples):
        gt_text = tokenizer.decode(x_text[i])
        pred_text = tokenizer.decode(predicted_ids[i].cpu())
        print(f"  GT:   {gt_text}")
        print(f"  Pred: {pred_text}")
        print()


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


def evaluate(
    workspace: GlobalWorkspace,
    eval_loader: DataLoader,
    config: dict,
    device: torch.device,
    tokenizer: Optional[SimpleTokenizer] = None,
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

        # Translation MSE (skip pour text car c'est des tokens)
        if target != "text":
            mse = compute_translation_mse(
                workspace, eval_loader, source, target, device
            )
            key = f"translation_mse_{source}→{target}"
            results[key] = mse
            print(f"  {key}: {mse:.6f}")

    # Traduction qualitative
    if tokenizer is not None:
        print()
        show_translation_examples(workspace, eval_loader, tokenizer, device)

    # Workspace stats
    stats = compute_workspace_stats(workspace, eval_loader, device)
    results.update(stats)
    print()
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    return results
