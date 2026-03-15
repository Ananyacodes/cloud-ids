from __future__ import annotations
import logging
from typing import List

import numpy as np
import torch

from ingestion.parsers import FlowRecord
from inference.serving.model_registry import ModelRegistry
from models.ensemble.arbitration import EnsembleArbitrator, EnsembleResult

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self) -> None:
        self.registry = ModelRegistry.get()
        self.arbitrator = EnsembleArbitrator()
        self._seq_len = self.registry.preprocessor.seq_len

    # Coordinates all three models on a batch of flows. The LSTM requires
    # fixed-length windows rather than single rows, so the tabular matrix is
    # converted to sliding windows and leading entries too short for a full
    # window are padded with 0.5 (neutral score).
    def predict_batch(self, records: List[FlowRecord]) -> List[EnsembleResult]:
        if not records:
            return []

        reg = self.registry
        device = reg.device

        import pandas as pd
        df = pd.DataFrame([r.to_dict() for r in records])
        X_tab = reg.preprocessor.transform(df)
        flow_ids   = [r.flow_id   for r in records]
        timestamps = [r.timestamp for r in records]

        xgb_scores = reg.xgb.predict_proba(X_tab)

        X_t = torch.tensor(X_tab, dtype=torch.float32).to(device)
        ae_scores = reg.autoencoder.predict_proba(X_t).cpu().numpy()

        X_seq = reg.preprocessor.make_sequences(X_tab)
        n_seq = len(X_seq)
        n_tab = len(X_tab)
        if n_seq < n_tab:
            pad = np.full(n_tab - n_seq, 0.5, dtype=np.float32)
            seq_t = torch.tensor(X_seq, dtype=torch.float32).to(device)
            lstm_raw = reg.lstm.predict_proba(seq_t).cpu().numpy()
            lstm_scores = np.concatenate([pad, lstm_raw])
        else:
            seq_t = torch.tensor(X_seq[:n_tab], dtype=torch.float32).to(device)
            lstm_scores = reg.lstm.predict_proba(seq_t).cpu().numpy()

        return self.arbitrator.arbitrate_batch(
            flow_ids, lstm_scores, xgb_scores, ae_scores, timestamps
        )

    def predict_one(self, record: FlowRecord) -> EnsembleResult:
        return self.predict_batch([record])[0]
