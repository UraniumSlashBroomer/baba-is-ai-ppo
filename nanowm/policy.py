from __future__ import annotations

import torch
import torch.nn as nn


class LatentActionHead(nn.Module):
    """Small supervised policy head over the current-frame VAE latent."""

    def __init__(self, latent_channels: int, latent_size: int, action_dim: int, hidden_size: int = 256) -> None:
        super().__init__()
        input_dim = int(latent_channels) * int(latent_size) * int(latent_size)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, current_latent: torch.Tensor) -> torch.Tensor:
        return self.net(current_latent)
