"""
decoders.py — Décodeurs domaine-spécifiques
=============================================
VisionDecoder : z_vision (128) → features ResNet-50 reconstruites (2048)
TextDecoder   : z_text (128) → embeddings E5 reconstruits (1024)

Les deux décodeurs utilisent MSE comme loss de reconstruction.
"""

import torch
import torch.nn as nn


class VisionDecoder(nn.Module):
    """z_vision → features ResNet-50 reconstruites.

    MLP 3 couches miroir de l'encodeur : 128 → 512 → 1024 → 2048.
    Loss Phase 1 : MSE(features_hat, features_original).
    """

    def __init__(self, latent_dim: int = 128, feature_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, feature_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → features_hat: (B, 2048)"""
        return self.net(z)


class TextDecoder(nn.Module):
    """z_text → embeddings E5 reconstruits.

    MLP simple : 128 → 512 → 1024.
    Loss Phase 1 : MSE(e5_hat, e5_original).
    """

    def __init__(self, latent_dim: int = 128, feature_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, feature_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → e5_hat: (B, 1024)"""
        return self.net(z)
