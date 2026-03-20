from __future__ import annotations
import json
import logging
import os

from google.cloud import pubsub_v1

from configs.loader import get_config
from models.ensemble.arbitration import Decision, EnsembleResult

logger = logging.getLogger(__name__)


class TriageRouter:
    def __init__(self) -> None:
        self._publisher = None
        self._alert_path = ""
        self._analyst_path = ""

        if str(os.environ.get("IDS_DISABLE_PUBSUB", "")).strip().lower() in {"1", "true", "yes", "on"}:
            logger.info("IDS_DISABLE_PUBSUB enabled; triage publishing disabled.")
            return

        cfg = get_config()
        self._project = cfg.gcp["project_id"]
        self._alert_topic = cfg.gcp["pubsub_alert_topic"]
        self._analyst_topic = cfg.gcp["pubsub_analyst_topic"]
        try:
            self._publisher = pubsub_v1.PublisherClient()
            self._alert_path = self._publisher.topic_path(self._project, self._alert_topic)
            self._analyst_path = self._publisher.topic_path(self._project, self._analyst_topic)
        except Exception as exc:
            logger.warning("Pub/Sub init failed; triage publishing disabled: %s", exc)
            self._publisher = None

    def route(self, result: EnsembleResult) -> bool:
        if self._publisher is None:
            return False

        payload = json.dumps({
            "flow_id": result.flow_id,
            "timestamp": result.timestamp,
            "ensemble_score": result.ensemble_score,
            "lstm_score": result.lstm_score,
            "xgb_score": result.xgb_score,
            "ae_score": result.ae_score,
            "decision": result.decision.value,
        }).encode()

        if result.decision == Decision.ALLOW:
            return False

        if result.decision == Decision.QUEUE:
            future = self._publisher.publish(self._analyst_path, payload)
            future.add_done_callback(self._log_publish_result)
            logger.debug("Queued flow %s for analyst review (score=%.3f)",
                         result.flow_id, result.ensemble_score)
            return True

        if result.decision == Decision.BLOCK:
            future = self._publisher.publish(
                self._alert_path, payload,
                severity="HIGH",
                flow_id=result.flow_id,
            )
            future.add_done_callback(self._log_publish_result)
            logger.warning("BLOCKED flow %s (score=%.3f)", result.flow_id, result.ensemble_score)
            return True

        return False

    @staticmethod
    def _log_publish_result(future) -> None:
        try:
            future.result()
        except Exception as exc:
            logger.error("Pub/Sub publish failed: %s", exc)
