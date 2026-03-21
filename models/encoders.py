"""
encoders.py — Encodeurs domaine-spécifiques
============================================
VisionEncoder : features ResNet-50 pré-extraites (2048) → z_vision
TextEncoder   : token IDs → z_text (Transformer + masked mean pooling)
"""

import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    """Features ResNet-50 pré-extraites → z_vision.

    Le ResNet est exécuté offline (scripts/extract_features.py).
    Ici on entraîne juste une tête linéaire sur les features 2048-dim.
    """

    def __init__(self, feature_dim: int = 2048, latent_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2048) features pré-extraites → z: (B, latent_dim)"""
        return self.net(x)


class TextEncoder(nn.Module):
    """Token IDs → z_text. Embedding + Transformer + masked mean pooling."""

    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 256,
        latent_dim: int = 512,
        max_seq_len: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(embed_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len) token ids → z: (B, latent_dim)"""
        B, S = x.shape
        padding_mask = x == 0  # True where padded

        positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.embedding(x) + self.pos_embedding(positions)
        h = self.transformer(h, src_key_padding_mask=padding_mask)

        # Masked mean pooling (ignore padding tokens)
        mask = (~padding_mask).unsqueeze(-1).float()  # (B, S, 1)
        h = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return self.proj(h)
