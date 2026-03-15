from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from configs.loader import get_config
from training.dataset import load_parquet, make_loader
from features.preprocessor import FeaturePreprocessor
from models.autoencoder.model import Autoencoder

logger = logging.getLogger(__name__)


def train(config_path: str | None = None) -> None:
    cfg = get_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_parquet(cfg.training["dataset_gcs_path"])

    preprocessor = FeaturePreprocessor()
    preprocessor.fit(df)

    benign_df = df[df["label"] == 0].copy()
    logger.info("Benign samples for AE training: %d", len(benign_df))
    X_benign = preprocessor.transform(benign_df)

    split = int(len(X_benign) * 0.9)
    X_train, X_thresh = X_benign[:split], X_benign[split:]

    loader = make_loader(X_train, np.zeros(len(X_train), dtype=np.int32),
                         batch_size=256, sequence=False)

    m_cfg = cfg.models["autoencoder"]
    model = Autoencoder(
        input_dim=m_cfg["input_dim"],
        hidden_dims=m_cfg["hidden_dims"],
        latent_dim=m_cfg["latent_dim"],
        dropout=m_cfg["dropout"],
    ).to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    patience_ctr = 0

    for epoch in range(1, cfg.training["max_epochs"] + 1):
        model.train()
        total_loss = 0.0
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            optimizer.zero_grad()
            recon = model(X_batch)
            loss = criterion(recon, X_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        logger.info("AE Epoch %d | loss=%.6f", epoch, avg)
        if avg < best_loss:
            best_loss = avg
            patience_ctr = 0
            out = Path("artifacts/autoencoder_best.pt")
            out.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), out)
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.training["early_stopping_patience"]:
                break

    model.load_state_dict(torch.load("artifacts/autoencoder_best.pt", map_location=device))
    model.eval()
    threshold = model.fit_threshold(X_thresh, m_cfg["reconstruction_threshold_percentile"])
    logger.info("Anomaly threshold set at %.6f (p%d)", threshold,
                m_cfg["reconstruction_threshold_percentile"])
    torch.save({"state_dict": model.state_dict(), "threshold": threshold},
               "artifacts/autoencoder_final.pt")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
