from __future__ import annotations
import logging
from pathlib import Path

from google.cloud import storage

from configs.loader import get_config

logger = logging.getLogger(__name__)

ARTIFACT_FILES = [
    "artifacts/lstm_best.pt",
    "artifacts/autoencoder_final.pt",
    "artifacts/xgboost_model.pkl",
    "artifacts/preprocessor.pkl",
]


def upload(config_path: str | None = None) -> None:
    cfg = get_config(config_path)
    bucket_name = cfg.gcp["gcs_bucket"]
    artifacts_prefix = cfg.models["artifacts_dir"].replace(f"gs://{bucket_name}/", "")

    client = storage.Client(project=cfg.gcp["project_id"])
    bucket = client.bucket(bucket_name)

    for local_path in ARTIFACT_FILES:
        p = Path(local_path)
        if not p.exists():
            logger.warning("Artifact not found, skipping: %s", local_path)
            continue
        blob_name = f"{artifacts_prefix}/{p.name}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(p))
        logger.info("Uploaded %s -> gs://%s/%s", local_path, bucket_name, blob_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    upload()
