"""
encoders.py — Encodeurs domaine-spécifiques
============================================
VisionEncoder : features ResNet-50 pré-extraites (2048) → z_vision (64)
TextEncoder   : embeddings E5-large-v2 pré-extraits (1024) → z_text (64)
BrainEncoder  : betas fMRI PCA-réduits (1024) → z_brain (64)
"""

import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    """Features ResNet-50 pré-extraites → z_vision.

    MLP 3 couches pour une compression progressive 2048 → 128.
    Le ResNet est exécuté offline (scripts/extract_features.py).
    """

    def __init__(self, feature_dim: int = 2048, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2048) features pré-extraites → z: (B, latent_dim)"""
        return self.net(x)


class TextEncoder(nn.Module):
    """Embeddings E5-large-v2 pré-extraits → z_text.

    MLP 2 couches : 1024 → 512 → 128.
    Les embeddings sont extraits offline (scripts/extract_text_features.py).
    """

    def __init__(self, feature_dim: int = 1024, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1024) E5 embeddings → z: (B, latent_dim)"""
        return self.net(x)


class BrainEncoder(nn.Module):
    """PCA-reduced fMRI betas → z_brain.

    MLP 3 couches : 1024 → 768 → 512 → latent_dim.
    Plus profond que TextEncoder car les betas fMRI sont plus bruitées
    et nécessitent plus de capacité pour extraire le signal discriminatif.
    Les betas sont pré-extraits offline (scripts/extract_brain_features.py).
    """

    def __init__(self, feature_dim: int = 1024, latent_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1024) PCA betas → z: (B, latent_dim)"""
        return self.net(x)
