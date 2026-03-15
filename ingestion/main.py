from __future__ import annotations
import logging

from ingestion.pubsub_consumer import PubSubConsumer
from ingestion.parsers import parse_message
from inference.serving.predictor import Predictor
from triage.router import TriageRouter
from monitoring.drift_detector import DriftDetector
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_predictor = Predictor()
_router = TriageRouter()
_drift = DriftDetector()
_benign_buffer = []


def handle_message(raw: dict) -> None:
    record = parse_message(raw)
    if record is None:
        return
    result = _predictor.predict_one(record)
    _router.route(result)

    from models.ensemble.arbitration import Decision
    if result.decision == Decision.ALLOW:
        _benign_buffer.append(record)
        if len(_benign_buffer) >= 1000:
            from features.preprocessor import FeaturePreprocessor
            import pandas as pd
            df = pd.DataFrame([r.to_dict() for r in _benign_buffer])
            preprocessor = _predictor.registry.preprocessor
            X = preprocessor.transform(df)
            if _drift.update(X):
                _drift.trigger_retraining()
            _benign_buffer.clear()


if __name__ == "__main__":
    consumer = PubSubConsumer(handler=handle_message)
    consumer.start()
