from __future__ import annotations
import logging
from pathlib import Path

import torch

from configs.loader import get_config
from models.lstm.model import LSTMClassifier
from models.autoencoder.model import Autoencoder
from models.xgboost.model import XGBoostClassifier
from features.preprocessor import FeaturePreprocessor

logger = logging.getLogger(__name__)


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


        pre_local = cache_dir / "preprocessor.pkl"
        if not pre_local.exists():
            _gcs_download(f"{artifacts_dir}/preprocessor.pkl", pre_local)
        self.preprocessor = FeaturePreprocessor().load(pre_local)


        xgb_local = cache_dir / "xgboost_model.pkl"
        if not xgb_local.exists():
            _gcs_download(f"{artifacts_dir}/xgboost_model.pkl", xgb_local)
        self.xgb = XGBoostClassifier()
        self.xgb.load(xgb_local)
        logger.info("XGBoost loaded")


        lstm_local = cache_dir / "lstm_best.pt"
        if not lstm_local.exists():
            _gcs_download(f"{artifacts_dir}/lstm_best.pt", lstm_local)
        m_cfg = cfg.models["lstm"]
        self.lstm = LSTMClassifier(
            input_size=m_cfg["input_size"],
            hidden_size=m_cfg["hidden_size"],
            num_layers=m_cfg["num_layers"],
            dropout=m_cfg["dropout"],
        )
        self.lstm.load_state_dict(torch.load(lstm_local, map_location=device))
        self.lstm.to(device).eval()
        logger.info("LSTM loaded")


        ae_local = cache_dir / "autoencoder_final.pt"
        if not ae_local.exists():
            _gcs_download(f"{artifacts_dir}/autoencoder_final.pt", ae_local)
        ae_ckpt = torch.load(ae_local, map_location=device)
        m_cfg_ae = cfg.models["autoencoder"]
        self.autoencoder = Autoencoder(
            input_dim=m_cfg_ae["input_dim"],
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
