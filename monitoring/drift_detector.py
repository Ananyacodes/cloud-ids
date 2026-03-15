from __future__ import annotations
import logging
from collections import deque

import numpy as np
import torch

from configs.loader import get_config
from inference.serving.model_registry import ModelRegistry
from monitoring.metrics import RECONSTRUCTION_ERROR_GAUGE

logger = logging.getLogger(__name__)


class DriftDetector:
    def __init__(self, window: int = 10_000) -> None:
        self._window: deque[float] = deque(maxlen=window)
        self._cfg = get_config()
        self._threshold = float(self._cfg.retraining["drift_threshold"])

    # Adds confirmed-benign reconstruction errors to a rolling window and checks
    # whether the mean has risen above the baseline threshold by more than the
    # configured drift fraction. Returns True if retraining should be triggered.
    def update(self, X_benign: np.ndarray) -> bool:
        reg = ModelRegistry.get()
        t = torch.tensor(X_benign, dtype=torch.float32).to(reg.device)
        errors = reg.autoencoder.reconstruction_error(t).cpu().numpy()
        self._window.extend(errors.tolist())

        mean_err = float(np.mean(self._window))
        RECONSTRUCTION_ERROR_GAUGE.set(mean_err)

        baseline = reg.autoencoder.threshold
        delta = (mean_err - baseline) / (baseline + 1e-9)

        if delta > self._threshold:
            logger.warning(
                "Drift detected: mean reconstruction error %.6f is %.1f%% above baseline %.6f",
                mean_err, delta * 100, baseline,
            )
            return True
        return False

    def trigger_retraining(self) -> None:
        logger.info("Triggering retraining job...")
        try:
            from google.cloud import scheduler_v1
            cfg = get_config()
            client = scheduler_v1.CloudSchedulerClient()
            job_name = f"projects/{cfg.gcp['project_id']}/locations/{cfg.gcp['region']}/jobs/ids-retrain"
            client.run_job(name=job_name)
            logger.info("Retraining job triggered: %s", job_name)
        except Exception as exc:
            logger.error("Failed to trigger retraining: %s", exc)
