from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin


class XGBoostClassifier:
    def __init__(self, **kwargs) -> None:
        defaults = dict(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            scale_pos_weight=5.0,
            eval_metric="aucpr",
            early_stopping_rounds=30,
            use_label_encoder=False,
            tree_method="hist",
            device="cpu",
            random_state=42,
        )
        defaults.update(kwargs)
        self._xgb = xgb.XGBClassifier(**defaults)
        self._calibrated: CalibratedClassifierCV | None = None

    # Two-phase training: fits XGBoost with early stopping, then applies Platt
    # scaling (sigmoid calibration) on the held-out validation set so that the
    # output of predict_proba is a well-calibrated probability, not a raw score.
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "XGBoostClassifier":
        self._xgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )
        self._calibrated = CalibratedClassifierCV(
            self._xgb, method="sigmoid", cv="prefit"
        )
        self._calibrated.fit(X_val, y_val)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        clf = self._calibrated if self._calibrated else self._xgb
        return clf.predict_proba(X)[:, 1]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"xgb": self._xgb, "calibrated": self._calibrated}, f)

    def load(self, path: str | Path) -> "XGBoostClassifier":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self._xgb = obj["xgb"]
        self._calibrated = obj["calibrated"]
        return self
