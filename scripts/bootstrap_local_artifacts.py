from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from configs.loader import get_config
from features.preprocessor import FeaturePreprocessor
from models.autoencoder.model import Autoencoder
from models.lstm.model import LSTMClassifier
from models.xgboost.model import XGBoostClassifier

logger = logging.getLogger(__name__)


def _make_synthetic_df(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    cfg = get_config()
    rng = np.random.default_rng(seed)

    protocols = np.array(["TCP", "UDP", "ICMP"])
    src_buckets = np.array(["well_known", "registered", "ephemeral"])
    dst_buckets = np.array(["well_known", "registered", "ephemeral", "unknown"])
    sources = np.array(["pcap", "netflow", "app_log", "tls"])

    rows: list[dict] = []
    for i in range(n):
        duration = float(rng.uniform(0.01, 10.0))
        bytes_fwd = int(rng.integers(40, 50000))
        bytes_bwd = int(rng.integers(40, 50000))
        packets_fwd = int(rng.integers(1, 200))
        packets_bwd = int(rng.integers(1, 200))
        total_bytes = bytes_fwd + bytes_bwd
        total_pkts = packets_fwd + packets_bwd

        syn = int(rng.integers(0, 6))
        rst = int(rng.integers(0, 3))
        is_attack = int((syn >= 4 and rst >= 1) or (total_bytes / duration > 25000))

        row = {
            "duration": duration,
            "bytes_fwd": bytes_fwd,
            "bytes_bwd": bytes_bwd,
            "packets_fwd": packets_fwd,
            "packets_bwd": packets_bwd,
            "pkt_size_mean": float(rng.uniform(60, 1500)),
            "pkt_size_std": float(rng.uniform(0, 400)),
            "iat_mean": float(rng.uniform(0.0001, 1.0)),
            "iat_std": float(rng.uniform(0.0, 0.5)),
            "fin_flag_cnt": int(rng.integers(0, 4)),
            "syn_flag_cnt": syn,
            "rst_flag_cnt": rst,
            "psh_flag_cnt": int(rng.integers(0, 8)),
            "ack_flag_cnt": int(rng.integers(0, 100)),
            "flow_bytes_per_sec": float(total_bytes / duration),
            "flow_pkts_per_sec": float(total_pkts / duration),
            "protocol": str(rng.choice(protocols)),
            "src_port_bucket": str(rng.choice(src_buckets)),
            "dst_port_bucket": str(rng.choice(dst_buckets)),
            "traffic_source": str(rng.choice(sources)),
            "label": is_attack,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    for col in cfg.numerical_cols + cfg.categorical_cols + ["label"]:
        if col not in df.columns:
            df[col] = 0

    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = get_config()
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)

    df = _make_synthetic_df()

    pre = FeaturePreprocessor().fit(df)
    X = pre.transform(df)
    y = df["label"].astype(np.int32).values
    feature_dim = int(X.shape[1])

    pre_path = artifacts / "preprocessor.pkl"
    pre.save(pre_path)
    logger.info("Wrote %s", pre_path)

    xgb_cfg = cfg.models["xgboost"]
    xgb = XGBoostClassifier(
        n_estimators=min(int(xgb_cfg.get("n_estimators", 200)), 80),
        max_depth=min(int(xgb_cfg.get("max_depth", 6)), 6),
        learning_rate=float(xgb_cfg.get("learning_rate", 0.1)),
        scale_pos_weight=float(xgb_cfg.get("scale_pos_weight", 1.0)),
        early_stopping_rounds=10,
    )
    split = int(len(X) * 0.8)
    xgb.fit(X[:split], y[:split], X[split:], y[split:])
    xgb_path = artifacts / "xgboost_model.pkl"
    xgb.save(xgb_path)
    logger.info("Wrote %s", xgb_path)

    lstm_cfg = cfg.models["lstm"]
    lstm = LSTMClassifier(
        input_size=feature_dim,
        hidden_size=int(lstm_cfg["hidden_size"]),
        num_layers=int(lstm_cfg["num_layers"]),
        dropout=float(lstm_cfg["dropout"]),
    )
    lstm_path = artifacts / "lstm_best.pt"
    torch.save(lstm.state_dict(), lstm_path)
    logger.info("Wrote %s", lstm_path)

    ae_cfg = cfg.models["autoencoder"]
    ae = Autoencoder(
        input_dim=feature_dim,
        hidden_dims=list(ae_cfg["hidden_dims"]),
        latent_dim=int(ae_cfg["latent_dim"]),
        dropout=float(ae_cfg["dropout"]),
    )
    ae.threshold = 0.05
    ae_path = artifacts / "autoencoder_final.pt"
    torch.save({"state_dict": ae.state_dict(), "threshold": ae.threshold}, ae_path)
    logger.info("Wrote %s", ae_path)

    logger.info("Bootstrap artifacts generated for offline local mode.")


if __name__ == "__main__":
    main()
