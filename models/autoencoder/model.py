from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def _build_encoder(input_dim: int, hidden_dims: list[int], latent_dim: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, latent_dim))
    return nn.Sequential(*layers)


def _build_decoder(latent_dim: int, hidden_dims: list[int], output_dim: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = latent_dim
    for h in reversed(hidden_dims):
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 21,
        hidden_dims: list[int] | None = None,
        latent_dim: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]
        self.encoder = _build_encoder(input_dim, hidden_dims, latent_dim, dropout)
        self.decoder = _build_decoder(latent_dim, hidden_dims, input_dim, dropout)
        self.threshold: float = 0.0

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruction_error(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            recon = self.forward(x)
            return ((recon - x) ** 2).mean(dim=1)

    # Calibrates the anomaly threshold using benign-only data. Takes the given
    # percentile of reconstruction errors as the decision boundary so that
    # predict_proba returns near-1 for inputs more anomalous than most benign traffic.
    def fit_threshold(self, benign_X: np.ndarray, percentile: float = 95.0) -> float:
        t = torch.tensor(benign_X, dtype=torch.float32)
        errors = self.reconstruction_error(t).numpy()
        self.threshold = float(np.percentile(errors, percentile))
        return self.threshold

    # Converts reconstruction error to an anomaly probability in [0, 1].
    # Uses a sigmoid centered at the threshold so benign samples score near 0
    # and anomalous samples score near 1. The scale factor sharpens the curve.
    def predict_proba(self, x: Tensor) -> Tensor:
        errors = self.reconstruction_error(x)
        scale = 1.0 / (self.threshold + 1e-8)
        return torch.sigmoid((errors - self.threshold) * scale * 5)
