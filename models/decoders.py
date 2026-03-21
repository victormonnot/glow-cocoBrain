"""
decoders.py — Décodeurs domaine-spécifiques
=============================================
VisionDecoder : z_vision → features ResNet-50 reconstruites (2048)
TextDecoder   : z_text → logits sur le vocabulaire (non-autorégressif)
"""

import torch
import torch.nn as nn


class VisionDecoder(nn.Module):
    """z_vision → features ResNet-50 reconstruites.

    On reconstruit les features 2048-dim, pas les pixels.
    Loss Phase 1 : MSE(features_hat, features_original).
    """

    def __init__(self, latent_dim: int = 512, feature_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, feature_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → features_hat: (B, 2048)"""
        return self.net(z)


class TextDecoder(nn.Module):
    """z_text → Logits sur le vocabulaire pour chaque position.

    Non-autorégressif : génère toute la séquence en parallèle.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        embed_dim: int = 256,
        vocab_size: int = 10000,
        max_seq_len: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.latent_to_seq = nn.Linear(latent_dim, max_seq_len * embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → logits: (B, seq_len, vocab_size)"""
        B = z.size(0)
        h = self.latent_to_seq(z).view(B, self.max_seq_len, -1)
        positions = torch.arange(self.max_seq_len, device=z.device).unsqueeze(0).expand(B, -1)
        h = h + self.pos_embedding(positions)
        h = self.transformer(h)
        return self.head(h)
