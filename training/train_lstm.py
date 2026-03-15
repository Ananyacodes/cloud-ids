from __future__ import annotations
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from configs.loader import get_config
from training.dataset import load_parquet, prepare_splits, make_loader
from features.preprocessor import FeaturePreprocessor
from models.lstm.model import LSTMClassifier
from training.metrics import compute_metrics

logger = logging.getLogger(__name__)


# Custom loss that penalises false negatives (missed attacks) more heavily than
# false positives. Each positive sample receives fn_weight times more gradient
# signal, which pushes the model to achieve high recall at the cost of precision.
def asymmetric_bce(
    logits: torch.Tensor, targets: torch.Tensor, fn_weight: float = 5.0
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    weight = targets * fn_weight + (1 - targets) * 1.0
    loss = nn.functional.binary_cross_entropy(probs, targets, weight=weight)
    return loss


def train(config_path: str | None = None) -> None:
    cfg = get_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training LSTM on %s", device)

    df = load_parquet(cfg.training["dataset_gcs_path"])
    preprocessor = FeaturePreprocessor()
    splits = prepare_splits(
        df, preprocessor,
        test_split=cfg.training["test_split"],
        val_split=cfg.training["val_split"],
        seed=cfg.training["random_seed"],
    )
    preprocessor.save("artifacts/preprocessor.pkl")

    seq_len = cfg.sequence_length
    train_loader = make_loader(*splits["train"], batch_size=cfg.training["batch_size"],
                               sequence=True, seq_len=seq_len)
    val_loader   = make_loader(*splits["val"],   batch_size=256, shuffle=False,
                               sequence=True, seq_len=seq_len)

    m_cfg = cfg.models["lstm"]
    model = LSTMClassifier(
        input_size=m_cfg["input_size"],
        hidden_size=m_cfg["hidden_size"],
        num_layers=m_cfg["num_layers"],
        dropout=m_cfg["dropout"],
    ).to(device)
    fn_weight = float(m_cfg["fn_fp_penalty_ratio"])

    optimizer = AdamW(model.parameters(), lr=cfg.training["learning_rate"], weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    best_val_f1 = 0.0
    patience_ctr = 0
    patience = cfg.training["early_stopping_patience"]

    for epoch in range(1, cfg.training["max_epochs"] + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = asymmetric_bce(logits, y_batch, fn_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                scores = model.predict_proba(X_batch.to(device)).cpu().numpy()
                all_scores.extend(scores)
                all_labels.extend(y_batch.numpy())

        metrics = compute_metrics(all_labels, all_scores)
        scheduler.step(metrics["f1"])
        logger.info(
            "Epoch %d | loss=%.4f | val_f1=%.4f | recall=%.4f | precision=%.4f",
            epoch, train_loss / len(train_loader),
            metrics["f1"], metrics["recall"], metrics["precision"],
        )

        if metrics["f1"] > best_val_f1:
            best_val_f1 = metrics["f1"]
            patience_ctr = 0
            out = Path("artifacts/lstm_best.pt")
            out.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), out)
            logger.info("  → Saved best model (f1=%.4f)", best_val_f1)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break


    model.load_state_dict(torch.load("artifacts/lstm_best.pt", map_location=device))
    test_loader = make_loader(*splits["test"], batch_size=256, shuffle=False,
                              sequence=True, seq_len=seq_len)
    all_scores, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            scores = model.predict_proba(X_batch.to(device)).cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(y_batch.numpy())
    test_metrics = compute_metrics(all_labels, all_scores)
    logger.info("TEST: %s", test_metrics)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
