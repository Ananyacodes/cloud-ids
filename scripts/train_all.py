from __future__ import annotations
import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all IDS models")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--skip-lstm",        action="store_true")
    parser.add_argument("--skip-xgboost",     action="store_true")
    parser.add_argument("--skip-autoencoder", action="store_true")
    args = parser.parse_args()

    os.environ["IDS_CONFIG"] = args.config
    Path("artifacts").mkdir(exist_ok=True)

    if not args.skip_xgboost:
        logger.info("=== Training XGBoost ===")
        from training.train_xgboost import train as train_xgb
        train_xgb(args.config)

    if not args.skip_lstm:
        logger.info("=== Training LSTM ===")
        from training.train_lstm import train as train_lstm
        train_lstm(args.config)

    if not args.skip_autoencoder:
        logger.info("=== Training Autoencoder ===")
        from training.train_autoencoder import train as train_ae
        train_ae(args.config)

    logger.info("=== All models trained. Uploading artifacts to GCS ===")
    from scripts.upload_artifacts import upload
    upload(args.config)


if __name__ == "__main__":
    main()
