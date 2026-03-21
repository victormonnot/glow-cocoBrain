"""
transforms.py — Preprocessing par domaine
===========================================
Simplifié pour cocoBrain : les features vision sont pré-extraites,
donc pas besoin de transforms image. Reste le padding texte.
"""

import torch


def text_transform(tokens: torch.Tensor, max_len: int = 32) -> torch.Tensor:
    """Pad ou tronque une séquence de tokens à max_len."""
    if tokens.size(0) > max_len:
        return tokens[:max_len]
    elif tokens.size(0) < max_len:
        padding = torch.zeros(max_len - tokens.size(0), dtype=tokens.dtype)
        return torch.cat([tokens, padding])
    return tokens
