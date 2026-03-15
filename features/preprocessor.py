from __future__ import annotations
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from configs.loader import get_config
from ingestion.parsers import FlowRecord


class FeaturePreprocessor:
    def __init__(self) -> None:
        cfg = get_config()
        self.num_cols = cfg.numerical_cols
        self.cat_cols = cfg.categorical_cols
        self.seq_len = cfg.sequence_length
        self._scaler: Optional[StandardScaler] = None
        self._encoders: dict[str, LabelEncoder] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "FeaturePreprocessor":
        for col in self.cat_cols:
            le = LabelEncoder()
            le.fit(df[col].fillna("unknown").astype(str))
            self._encoders[col] = le

        num_array = self._encode_categoricals(df)
        self._scaler = StandardScaler()
        self._scaler.fit(num_array)
        self._fitted = True
        return self


    def transform(self, df: pd.DataFrame) -> np.ndarray:
        assert self._fitted, "Call fit() first"
        encoded = self._encode_categoricals(df)
        return self._scaler.transform(encoded).astype(np.float32)

    def transform_record(self, record: FlowRecord) -> np.ndarray:
        row = pd.DataFrame([record.to_dict()])
        return self.transform(row)[0]

    # Creates fixed-length overlapping windows from a feature matrix for LSTM input.
    # Pads with zeros when the input has fewer rows than the required sequence length.
    # Uses numpy stride tricks to avoid copying data; output shape is (N-T+1, T, D).
    def make_sequences(self, X: np.ndarray) -> np.ndarray:
        T, D = self.seq_len, X.shape[1]
        if len(X) < T:
            pad = np.zeros((T - len(X), D), dtype=np.float32)
            X = np.vstack([pad, X])
        seqs = np.lib.stride_tricks.sliding_window_view(X, (T, D))
        return seqs.reshape(-1, T, D)


    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"scaler": self._scaler, "encoders": self._encoders}, f)

    def load(self, path: str | Path) -> "FeaturePreprocessor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self._scaler = obj["scaler"]
        self._encoders = obj["encoders"]
        self._fitted = True
        return self

    # Assembles a float32 array by stacking numerical columns (raw values) with
    # label-encoded categorical columns. Unknown category values are mapped to 0.
    def _encode_categoricals(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        for col in self.num_cols:
            parts.append(df[col].fillna(0).values.reshape(-1, 1))
        for col in self.cat_cols:
            le = self._encoders.get(col)
            raw = df[col].fillna("unknown").astype(str)
            if le is not None:
                encoded = np.array([
                    le.transform([v])[0] if v in le.classes_ else 0
                    for v in raw
                ])
            else:
                encoded = np.zeros(len(raw))
            parts.append(encoded.reshape(-1, 1))
        return np.hstack(parts).astype(np.float32)
