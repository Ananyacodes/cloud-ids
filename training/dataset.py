from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch

from configs.loader import get_config
from features.preprocessor import FeaturePreprocessor

logger = logging.getLogger(__name__)


def load_parquet(path: str) -> pd.DataFrame:
    if path.startswith("gs://"):
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        with fs.open(path) as f:
            return pd.read_parquet(f)
    return pd.read_parquet(path)


# Stratified train/val/test split with optional SMOTE oversampling.
# The val fraction is re-computed relative to the non-test portion so that
# val_split and test_split are both fractions of the total dataset size.
def prepare_splits(
    df: pd.DataFrame,
    preprocessor: FeaturePreprocessor,
    test_split: float = 0.15,
    val_split: float = 0.15,
    seed: int = 42,
    apply_smote: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    labels = df["label"].values
    preprocessor.fit(df)
    X = preprocessor.transform(df)
    y = labels.astype(np.int32)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_split, random_state=seed, stratify=y
    )
    val_frac = val_split / (1 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, random_state=seed, stratify=y_temp
    )

    if apply_smote:
        logger.info("Applying SMOTE. Train class distribution before: %s",
                    np.bincount(y_train))
        sm = SMOTE(random_state=seed, k_neighbors=5)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        logger.info("After SMOTE: %s", np.bincount(y_train))

    return {
        "train": (X_train.astype(np.float32), y_train),
        "val":   (X_val.astype(np.float32),   y_val),
        "test":  (X_test.astype(np.float32),  y_test),
    }


class FlowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, sequence: bool = False,
                 seq_len: int = 32) -> None:
        self.sequence = sequence
        self.seq_len = seq_len
        if sequence:
            T, D = seq_len, X.shape[1]
            if len(X) < T:
                pad = np.zeros((T - len(X), D), dtype=np.float32)
                X = np.vstack([pad, X])
                y = np.concatenate([np.zeros(T - len(y), dtype=np.int32), y])
            self.X = torch.tensor(
                np.lib.stride_tricks.sliding_window_view(X, (T, D)).reshape(-1, T, D),
                dtype=torch.float32,
            )
            self.y = torch.tensor(y[seq_len - 1:], dtype=torch.float32)
        else:
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int = 256,
                shuffle: bool = True, sequence: bool = False,
                seq_len: int = 32) -> DataLoader:
    ds = FlowDataset(X, y, sequence=sequence, seq_len=seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=4, pin_memory=True)
