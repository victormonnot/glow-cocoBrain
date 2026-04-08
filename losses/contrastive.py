"""
contrastive.py — Loss contrastive multi-positifs
==================================================
InfoNCE symétrique avec support des multi-positifs COCO.

Sur COCO, une image a 5 captions. Quand plusieurs captions de la
même image se retrouvent dans le même batch, l'InfoNCE standard
les traite comme des négatifs (faux négatifs). Ce fix utilise des
soft labels : pour chaque anchor, la target est une distribution
uniforme sur tous les positifs (même image_id), pas un one-hot
sur la diagonale.

Si image_ids n'est pas fourni, fallback sur l'InfoNCE standard
(diagonale = seul positif).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ContrastiveLoss(nn.Module):
    """InfoNCE symétrique avec multi-positifs."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        w_a: torch.Tensor,
        w_b: torch.Tensor,
        image_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            w_a: (B, D) — workspace repr du domaine A
            w_b: (B, D) — workspace repr du domaine B
            image_ids: (B,) — image IDs pour construire le masque multi-positifs.
                       Si None, fallback sur diagonale (1 positif par anchor).

        Returns:
            loss scalaire, mean_positives_per_anchor (pour logging)
        """
        # Normalise (cosine similarity)
        w_a = F.normalize(w_a, dim=-1)
        w_b = F.normalize(w_b, dim=-1)

        # Matrice de similarité (B, B)
        logits = torch.matmul(w_a, w_b.T) / self.temperature

        if image_ids is None:
            # Fallback : InfoNCE standard (diagonale)
            labels = torch.arange(logits.size(0), device=logits.device)
            loss_ab = F.cross_entropy(logits, labels)
            loss_ba = F.cross_entropy(logits.T, labels)
            return (loss_ab + loss_ba) / 2

        # Multi-positifs : soft labels basées sur image_id
        # pos_mask[i][j] = True si image_ids[i] == image_ids[j]
        pos_mask = image_ids.unsqueeze(1) == image_ids.unsqueeze(0)  # (B, B)

        # Target = distribution uniforme sur les positifs
        pos_counts = pos_mask.float().sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        targets = pos_mask.float() / pos_counts  # (B, B)

        # Cross-entropy avec soft labels : -sum(target * log_softmax(logits))
        log_probs_ab = F.log_softmax(logits, dim=1)
        log_probs_ba = F.log_softmax(logits.T, dim=1)

        loss_ab = -(targets * log_probs_ab).sum(dim=1).mean()
        loss_ba = -(targets * log_probs_ba).sum(dim=1).mean()

        return (loss_ab + loss_ba) / 2
