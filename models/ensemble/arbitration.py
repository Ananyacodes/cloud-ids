from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from configs.loader import get_config


class Decision(str, Enum):
    ALLOW = "allow"
    QUEUE = "queue"
    BLOCK = "block"


@dataclass
class EnsembleResult:
    flow_id: str
    lstm_score: float
    xgb_score: float
    ae_score: float
    ensemble_score: float
    decision: Decision
    timestamp: str = ""


class EnsembleArbitrator:
    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
    ) -> None:
        cfg = get_config()
        self._weights = weights or cfg.models["ensemble"]["weights"]
        self._low = cfg.low_threshold
        self._high = cfg.high_threshold
        total = sum(self._weights.values())
        self._w = {k: v / total for k, v in self._weights.items()}

    def arbitrate(
        self,
        flow_id: str,
        lstm_score: float,
        xgb_score: float,
        ae_score: float,
        timestamp: str = "",
    ) -> EnsembleResult:
        ensemble = (
            self._w["lstm"] * lstm_score
            + self._w["xgboost"] * xgb_score
            + self._w["autoencoder"] * ae_score
        )
        decision = self._decide(ensemble)
        return EnsembleResult(
            flow_id=flow_id,
            lstm_score=lstm_score,
            xgb_score=xgb_score,
            ae_score=ae_score,
            ensemble_score=float(ensemble),
            decision=decision,
            timestamp=timestamp,
        )

    # Vectorized batch scoring: computes the weighted ensemble score for all
    # flows at once using NumPy array ops, then maps each scalar score to a
    # Decision by comparing against the low and high thresholds.
    def arbitrate_batch(
        self,
        flow_ids: list[str],
        lstm_scores: np.ndarray,
        xgb_scores: np.ndarray,
        ae_scores: np.ndarray,
        timestamps: Optional[list[str]] = None,
    ) -> list[EnsembleResult]:
        ensemble = (
            self._w["lstm"] * lstm_scores
            + self._w["xgboost"] * xgb_scores
            + self._w["autoencoder"] * ae_scores
        )
        ts_list = timestamps or [""] * len(flow_ids)
        results = []
        for i, fid in enumerate(flow_ids):
            results.append(EnsembleResult(
                flow_id=fid,
                lstm_score=float(lstm_scores[i]),
                xgb_score=float(xgb_scores[i]),
                ae_score=float(ae_scores[i]),
                ensemble_score=float(ensemble[i]),
                decision=self._decide(float(ensemble[i])),
                timestamp=ts_list[i],
            ))
        return results

    def _decide(self, score: float) -> Decision:
        if score < self._low:
            return Decision.ALLOW
        if score >= self._high:
            return Decision.BLOCK
        return Decision.QUEUE
