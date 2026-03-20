from __future__ import annotations

import argparse


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cloud-ids",
        description="Cloud IDS command-line interface",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    interfaces_parser = subparsers.add_parser(
        "interfaces",
        help="List local network interfaces available for live capture",
    )
    interfaces_parser.set_defaults(handler=_run_interfaces)

    scan_parser = subparsers.add_parser(
        "scan",
        help="Run live network scanning",
    )
    scan_parser.add_argument("--iface", default=None, help="Network interface name")
    scan_parser.add_argument("--bpf", default="ip", help="BPF filter")
    scan_parser.add_argument(
        "--flow-idle-timeout",
        type=float,
        default=5.0,
        help="Flush and score flow after idle seconds",
    )
    scan_parser.add_argument(
        "--mode",
        choices=["local", "dry-run", "publish", "shadow"],
        default="local",
        help="Execution mode",
    )
    scan_parser.add_argument("--pubsub-topic", default=None, help="Override Pub/Sub topic")
    scan_parser.add_argument("--project-id", default=None, help="Override GCP project id")
    scan_parser.set_defaults(handler=_run_scan)

    return parser


def _run_interfaces(args: argparse.Namespace) -> None:
    from scapy.all import get_if_list

    for iface in get_if_list():
        print(iface)


def _run_scan(args: argparse.Namespace) -> None:
    from scripts.scan_live import main as scan_main

    scan_args: list[str] = []
    if args.iface:
        scan_args.extend(["--iface", args.iface])
    if args.bpf:
        scan_args.extend(["--bpf", args.bpf])
    scan_args.extend(["--flow-idle-timeout", str(args.flow_idle_timeout)])

    if args.mode == "dry-run":
        scan_args.append("--dry-run")
    elif args.mode == "publish":
        scan_args.append("--publish-pubsub")
    elif args.mode == "shadow":
        scan_args.append("--shadow-pubsub")

    if args.pubsub_topic:
        scan_args.extend(["--pubsub-topic", args.pubsub_topic])
    if args.project_id:
        scan_args.extend(["--project-id", args.project_id])

    scan_main(scan_args)


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
