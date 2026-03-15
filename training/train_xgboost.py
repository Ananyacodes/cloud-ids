from __future__ import annotations
import logging
from pathlib import Path

from configs.loader import get_config
from training.dataset import load_parquet, prepare_splits
from features.preprocessor import FeaturePreprocessor
from models.xgboost.model import XGBoostClassifier
from training.metrics import compute_metrics

logger = logging.getLogger(__name__)


def train(config_path: str | None = None) -> None:
    cfg = get_config(config_path)
    df = load_parquet(cfg.training["dataset_gcs_path"])
    preprocessor = FeaturePreprocessor()
    splits = prepare_splits(df, preprocessor,
                            test_split=cfg.training["test_split"],
                            val_split=cfg.training["val_split"],
                            seed=cfg.training["random_seed"])
    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]
    X_test,  y_test  = splits["test"]

    m_cfg = cfg.models["xgboost"]
    model = XGBoostClassifier(
        n_estimators=m_cfg["n_estimators"],
        max_depth=m_cfg["max_depth"],
        learning_rate=m_cfg["learning_rate"],
        scale_pos_weight=m_cfg["scale_pos_weight"],
        early_stopping_rounds=m_cfg["early_stopping_rounds"],
    )
    model.fit(X_train, y_train, X_val, y_val)

    out = Path("artifacts/xgboost_model.pkl")
    out.parent.mkdir(exist_ok=True)
    model.save(out)

    test_scores = model.predict_proba(X_test)
    logger.info("TEST: %s", compute_metrics(y_test, test_scores))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
