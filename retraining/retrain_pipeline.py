from __future__ import annotations
import io
import json
import logging
from pathlib import Path

import pandas as pd
from google.cloud import storage

from configs.loader import get_config

logger = logging.getLogger(__name__)


def load_analyst_verdicts(cfg) -> pd.DataFrame:
    client = storage.Client(project=cfg.gcp["project_id"])
    bucket_name = cfg.gcp["gcs_bucket"]
    prefix = cfg.retraining["verdict_gcs_path"].replace(f"gs://{bucket_name}/", "")
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    if not blobs:
        logger.warning("No analyst verdicts found at %s", cfg.retraining["verdict_gcs_path"])
        return pd.DataFrame()

    records = []
    for blob in blobs:
        data = json.loads(blob.download_as_text())
        records.extend(data if isinstance(data, list) else [data])
    df = pd.DataFrame(records)
    logger.info("Loaded %d analyst verdicts", len(df))
    return df


# Appends high-confidence analyst verdicts (confidence >= 0.8) to the base
# training parquet. Only columns present in both dataframes are kept to prevent
# schema mismatches when verdict fields differ from training data fields.
def merge_with_training_data(
    verdicts: pd.DataFrame, base_path: str
) -> pd.DataFrame:
    base_df = pd.read_parquet(base_path)
    if "confidence" in verdicts.columns:
        verdicts = verdicts[verdicts["confidence"] >= 0.8]
    if verdicts.empty:
        return base_df
    common_cols = list(set(base_df.columns) & set(verdicts.columns))
    merged = pd.concat([base_df, verdicts[common_cols]], ignore_index=True)
    logger.info("Merged dataset size: %d (was %d)", len(merged), len(base_df))
    return merged


def run(config_path: str | None = None) -> None:
    cfg = get_config(config_path)
    verdicts = load_analyst_verdicts(cfg)

    if verdicts.empty:
        logger.info("No new verdicts — skipping retraining")
        return

    base_path = cfg.training["dataset_gcs_path"]
    merged_df = merge_with_training_data(verdicts, base_path)

    local_merged = Path("/tmp/merged_dataset.parquet")
    merged_df.to_parquet(local_merged)

    import os
    os.environ["IDS_MERGED_DATASET"] = str(local_merged)

    from features.preprocessor import FeaturePreprocessor
    from training.dataset import prepare_splits
    from training.train_xgboost import train as train_xgb
    from training.train_lstm import train as train_lstm
    from training.train_autoencoder import train as train_ae
    from scripts.upload_artifacts import upload

    cfg.training["dataset_gcs_path"] = str(local_merged)

    logger.info("=== Retraining XGBoost ===")
    train_xgb()
    logger.info("=== Retraining LSTM ===")
    train_lstm()
    logger.info("=== Retraining Autoencoder ===")
    train_ae()

    logger.info("=== Uploading new artifacts ===")
    upload()

    from inference.serving.model_registry import ModelRegistry
    ModelRegistry._instance = None
    logger.info("Retraining complete. Model registry reset.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
