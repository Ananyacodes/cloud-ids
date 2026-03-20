from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ingestion.live_scanner import LivePacketScanner
from ingestion.parsers import parse_message

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _resolve_pubsub_target(project_override: str | None, topic_override: str | None) -> tuple[str, str]:
    if project_override and topic_override:
        return project_override, topic_override

    try:
        from configs.loader import get_config

        cfg = get_config()
        project_id = project_override or cfg.gcp["project_id"]
        topic_name = topic_override or cfg.gcp["pubsub_ingestion_topic"]
        return project_id, topic_name
    except Exception:
        import yaml

        config_path = Path("configs/config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        gcp_cfg = raw.get("gcp", {})
        project_id = project_override or gcp_cfg.get("project_id", "")
        topic_name = topic_override or gcp_cfg.get("pubsub_ingestion_topic", "")
        return project_id, topic_name


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan live network traffic and run IDS inference")
    parser.add_argument("--iface", default=None, help="Network interface name. Uses default if omitted.")
    parser.add_argument(
        "--bpf",
        default="ip",
        help="BPF filter for capture (examples: 'tcp', 'udp port 53', 'host 10.0.0.1').",
    )
    parser.add_argument(
        "--flow-idle-timeout",
        type=float,
        default=5.0,
        help="Flush and score a flow when idle for this many seconds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only parse and print extracted flow features without model inference.",
    )
    parser.add_argument(
        "--publish-pubsub",
        action="store_true",
        help="Publish captured flow messages to Pub/Sub ingestion topic instead of local inference.",
    )
    parser.add_argument(
        "--shadow-pubsub",
        action="store_true",
        help="Run local inference and also publish captured flow messages to Pub/Sub.",
    )
    parser.add_argument(
        "--pubsub-topic",
        default=None,
        help="Override Pub/Sub topic name (defaults to gcp.pubsub_ingestion_topic from config).",
    )
    parser.add_argument(
        "--project-id",
        default=None,
        help="Override GCP project id (defaults to gcp.project_id from config).",
    )
    args = parser.parse_args(argv)
    if args.dry_run and (args.publish_pubsub or args.shadow_pubsub):
        parser.error("--dry-run cannot be used with --publish-pubsub or --shadow-pubsub")
    if args.publish_pubsub and args.shadow_pubsub:
        parser.error("--publish-pubsub and --shadow-pubsub are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.dry_run:

        def on_flow(msg: dict) -> None:
            record = parse_message(msg)
            if record is None:
                logger.warning("Dropped flow: unsupported source=%s", msg.get("source"))
                return
            logger.info(
                "flow=%s src=%s proto=%s dur=%.2fs bytes=%d pkts=%d",
                record.flow_id[:10],
                record.traffic_source,
                record.protocol,
                record.duration,
                record.bytes_fwd + record.bytes_bwd,
                record.packets_fwd + record.packets_bwd,
            )

    elif args.publish_pubsub:
        from google.cloud import pubsub_v1

        project_id, topic_name = _resolve_pubsub_target(args.project_id, args.pubsub_topic)

        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(project_id, topic_name)
        logger.info("Publishing captured flows to %s", topic_path)

        def _log_publish_result(future) -> None:
            try:
                msg_id = future.result()
                logger.debug("Published flow message id=%s", msg_id)
            except Exception as exc:
                logger.error("Pub/Sub publish failed: %s", exc)

        def on_flow(msg: dict) -> None:
            payload = json.dumps(msg, separators=(",", ":")).encode("utf-8")
            future = publisher.publish(topic_path, payload)
            future.add_done_callback(_log_publish_result)

    elif args.shadow_pubsub:
        from google.cloud import pubsub_v1
        from ingestion.main import handle_message

        project_id, topic_name = _resolve_pubsub_target(args.project_id, args.pubsub_topic)

        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(project_id, topic_name)
        logger.info("Shadow mode enabled: local inference + publish to %s", topic_path)

        def _log_publish_result(future) -> None:
            try:
                msg_id = future.result()
                logger.debug("Published flow message id=%s", msg_id)
            except Exception as exc:
                logger.error("Pub/Sub publish failed: %s", exc)

        def on_flow(msg: dict) -> None:
            result = handle_message(msg)
            if result is not None:
                logger.info(
                    "flow=%s score=%.3f decision=%s",
                    result.flow_id[:10],
                    result.ensemble_score,
                    result.decision.value,
                )
            payload = json.dumps(msg, separators=(",", ":")).encode("utf-8")
            future = publisher.publish(topic_path, payload)
            future.add_done_callback(_log_publish_result)

    else:
        from ingestion.main import handle_message

        def on_flow(msg: dict) -> None:
            result = handle_message(msg)
            if result is not None:
                logger.info(
                    "flow=%s score=%.3f decision=%s",
                    result.flow_id[:10],
                    result.ensemble_score,
                    result.decision.value,
                )

    scanner = LivePacketScanner(
        on_flow=on_flow,
        interface=args.iface,
        bpf_filter=args.bpf,
        flow_idle_timeout=args.flow_idle_timeout,
    )

    logger.info("Live scan started. Press Ctrl+C to stop.")
    try:
        scanner.start()
    except KeyboardInterrupt:
        logger.info("Stopping on Ctrl+C")
        scanner.stop()


if __name__ == "__main__":
    main()
