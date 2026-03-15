from __future__ import annotations
import json
import logging
import signal
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from google.cloud import pubsub_v1

from configs.loader import get_config

logger = logging.getLogger(__name__)


class PubSubConsumer:
    def __init__(self, handler: Callable[[dict], None]) -> None:
        cfg = get_config()
        self._project = cfg.gcp["project_id"]
        self._subscription = cfg.gcp["pubsub_ingestion_sub"]
        self._handler = handler
        self._client = pubsub_v1.SubscriberClient()
        self._sub_path = self._client.subscription_path(
            self._project, self._subscription
        )
        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=8)

    def _on_message(self, message: pubsub_v1.types.ReceivedMessage) -> None:
        try:
            payload = json.loads(message.data.decode("utf-8"))
            self._handler(payload)
            message.ack()
        except Exception as exc:
            logger.error("Failed to process message: %s", exc)
            message.nack()

    def start(self) -> None:
        self._running = True
        flow_control = pubsub_v1.types.FlowControl(max_messages=512)
        self._future = self._client.subscribe(
            self._sub_path,
            callback=self._on_message,
            flow_control=flow_control,
            scheduler=pubsub_v1.subscriber.scheduler.ThreadScheduler(self._executor),
        )
        logger.info("Subscribed to %s", self._sub_path)

        def _shutdown(signum, frame):
            logger.info("Shutting down consumer…")
            self._future.cancel()

        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)

        try:
            self._future.result()
        except Exception as exc:
            logger.error("Subscriber exited: %s", exc)

    def stop(self) -> None:
        self._future.cancel()
        self._executor.shutdown(wait=False)
