from __future__ import annotations
import logging
import os
from pathlib import Path
import shutil

import torch

from configs.loader import get_config
from models.lstm.model import LSTMClassifier
from models.autoencoder.model import Autoencoder
from models.xgboost.model import XGBoostClassifier
from features.preprocessor import FeaturePreprocessor

logger = logging.getLogger(__name__)


def _is_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _find_local_artifact(artifacts_dir: str, filename: str) -> Path | None:
    candidates = [
        Path(artifacts_dir) / filename,
        Path("artifacts") / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _materialize_artifact(*, artifacts_dir: str, cache_dir: Path, filename: str) -> Path:
    cache_path = cache_dir / filename
    refresh_cache = _is_truthy(os.environ.get("IDS_REFRESH_LOCAL_ARTIFACTS"))
    if cache_path.exists() and not refresh_cache:
        return cache_path

    local_artifact = _find_local_artifact(artifacts_dir, filename)
    disable_gcs = _is_truthy(os.environ.get("IDS_DISABLE_GCS"))

    if local_artifact is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_artifact, cache_path)
        logger.info("Loaded local artifact %s -> %s", local_artifact, cache_path)
        return cache_path

    if artifacts_dir.startswith("gs://") and not disable_gcs:
        _gcs_download(f"{artifacts_dir}/{filename}", cache_path)
        return cache_path

    if artifacts_dir.startswith("gs://") and disable_gcs:
        raise FileNotFoundError(
            f"Missing local artifact {filename}. Set local files under artifacts/ or disable IDS_DISABLE_GCS."
        )

    raise FileNotFoundError(
        f"Artifact not found: {filename}. Checked {Path(artifacts_dir) / filename} and artifacts/{filename}."
    )


def _gcs_download(gcs_uri: str, local_path: Path) -> None:
    from google.cloud import storage
    cfg = get_config()
    bucket_name = gcs_uri.replace("gs://", "").split("/")[0]
    blob_path = "/".join(gcs_uri.replace("gs://", "").split("/")[1:])
    client = storage.Client(project=cfg.gcp["project_id"])
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))
    logger.info("Downloaded %s -> %s", gcs_uri, local_path)


class ModelRegistry:
    _instance: "ModelRegistry | None" = None

    def __init__(self) -> None:
        cfg = get_config()
        artifacts_dir = cfg.models["artifacts_dir"]
        cache_dir = Path(cfg.models["local_cache_dir"])
        cache_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        pre_local = _materialize_artifact(
            artifacts_dir=artifacts_dir,
            cache_dir=cache_dir,
            filename="preprocessor.pkl",
        )
        self.preprocessor = FeaturePreprocessor().load(pre_local)
        feature_dim = int(self.preprocessor._scaler.mean_.shape[0])

        xgb_local = _materialize_artifact(
            artifacts_dir=artifacts_dir,
            cache_dir=cache_dir,
            filename="xgboost_model.pkl",
        )
        self.xgb = XGBoostClassifier()
        self.xgb.load(xgb_local)
        logger.info("XGBoost loaded")

        lstm_local = _materialize_artifact(
            artifacts_dir=artifacts_dir,
            cache_dir=cache_dir,
            filename="lstm_best.pt",
        )
        m_cfg = cfg.models["lstm"]
        self.lstm = LSTMClassifier(
            input_size=feature_dim,
            hidden_size=m_cfg["hidden_size"],
            num_layers=m_cfg["num_layers"],
            dropout=m_cfg["dropout"],
        )
        self.lstm.load_state_dict(torch.load(lstm_local, map_location=device))
        self.lstm.to(device).eval()
        logger.info("LSTM loaded")

        ae_local = _materialize_artifact(
            artifacts_dir=artifacts_dir,
            cache_dir=cache_dir,
            filename="autoencoder_final.pt",
        )
        ae_ckpt = torch.load(ae_local, map_location=device)
        m_cfg_ae = cfg.models["autoencoder"]
        self.autoencoder = Autoencoder(
            input_dim=feature_dim,
            hidden_dims=m_cfg_ae["hidden_dims"],
            latent_dim=m_cfg_ae["latent_dim"],
            dropout=m_cfg_ae["dropout"],
        )
        self.autoencoder.load_state_dict(ae_ckpt["state_dict"])
        self.autoencoder.threshold = float(ae_ckpt["threshold"])
        self.autoencoder.to(device).eval()
        logger.info("Autoencoder loaded (threshold=%.6f)", self.autoencoder.threshold)

    @classmethod
    def get(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = ModelRegistry()
        return cls._instance
