from __future__ import annotations
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FlowRecord:
    flow_id: str
    timestamp: str
    traffic_source: str

    duration: float = 0.0
    bytes_fwd: int = 0
    bytes_bwd: int = 0
    packets_fwd: int = 0
    packets_bwd: int = 0

    pkt_size_mean: float = 0.0
    pkt_size_std: float = 0.0
    iat_mean: float = 0.0
    iat_std: float = 0.0

    fin_flag_cnt: int = 0
    syn_flag_cnt: int = 0
    rst_flag_cnt: int = 0
    psh_flag_cnt: int = 0
    ack_flag_cnt: int = 0

    flow_bytes_per_sec: float = 0.0
    flow_pkts_per_sec: float = 0.0

    protocol: str = "TCP"
    src_port_bucket: str = "ephemeral"
    dst_port_bucket: str = "unknown"

    ja3_hash: str = ""
    ja3s_hash: str = ""
    sni_entropy: float = 0.0
    cert_validity_days: int = -1
    tls_version: str = ""

    label: int = -1

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


def _port_bucket(port: int) -> str:
    if port < 1024:
        return "well_known"
    if port < 49152:
        return "registered"
    return "ephemeral"


def _safe_rate(numerator: float, duration: float) -> float:
    return numerator / duration if duration > 0 else 0.0


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# Produces a stable 16-character hex ID from the network 5-tuple. When none of
# the tuple fields are present it falls back to an MD5 of the full dict with
# sorted keys, ensuring the same logical flow always maps to the same ID.
def _derive_flow_id(data: dict[str, Any], protocol_hint: str = "") -> str:
    src_ip = str(data.get("src_ip", ""))
    dst_ip = str(data.get("dst_ip", ""))
    src_port = _to_int(data.get("src_port", 0))
    dst_port = _to_int(data.get("dst_port", data.get("port", 0)))
    proto = str(data.get("proto") or data.get("protocol") or protocol_hint or "").upper()

    if src_ip or dst_ip or src_port or dst_port or proto:
        material = f"{src_ip}|{src_port}|{dst_ip}|{dst_port}|{proto}"
    else:
        material = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)

    return hashlib.md5(material.encode()).hexdigest()[:16]


# Computes Shannon entropy (bits per character) of a string. High entropy in a
# TLS SNI hostname is a strong signal of DGA-generated or tunneling traffic.
def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {c: s.count(c) / len(s) for c in set(s)}
    return -sum(p * math.log2(p) for p in freq.values() if p > 0)


class PcapParser:
    def parse(self, raw: dict) -> FlowRecord:
        d = raw.get("data", {})
        dur = float(d.get("duration", 0))
        total_pkts = int(d.get("total_packets", 0))
        total_bytes = int(d.get("total_bytes", 0))
        src_port = int(d.get("src_port", 0))
        dst_port = int(d.get("dst_port", 0))
        flow_id = d.get("flow_id") or _derive_flow_id(d, protocol_hint=d.get("protocol", "TCP"))
        return FlowRecord(
            flow_id=flow_id,
            timestamp=raw.get("timestamp", ""),
            traffic_source="pcap",
            duration=dur,
            bytes_fwd=int(d.get("bytes_fwd", total_bytes // 2)),
            bytes_bwd=int(d.get("bytes_bwd", total_bytes // 2)),
            packets_fwd=int(d.get("packets_fwd", total_pkts // 2)),
            packets_bwd=int(d.get("packets_bwd", total_pkts // 2)),
            pkt_size_mean=float(d.get("pkt_size_mean", 0)),
            pkt_size_std=float(d.get("pkt_size_std", 0)),
            iat_mean=float(d.get("iat_mean", 0)),
            iat_std=float(d.get("iat_std", 0)),
            fin_flag_cnt=int(d.get("fin_flag_cnt", 0)),
            syn_flag_cnt=int(d.get("syn_flag_cnt", 0)),
            rst_flag_cnt=int(d.get("rst_flag_cnt", 0)),
            psh_flag_cnt=int(d.get("psh_flag_cnt", 0)),
            ack_flag_cnt=int(d.get("ack_flag_cnt", 0)),
            flow_bytes_per_sec=_safe_rate(total_bytes, dur),
            flow_pkts_per_sec=_safe_rate(total_pkts, dur),
            protocol=d.get("protocol", "TCP"),
            src_port_bucket=_port_bucket(src_port),
            dst_port_bucket=_port_bucket(dst_port),
        )


class NetFlowParser:
    def parse(self, raw: dict) -> FlowRecord:
        d = raw.get("data", {})
        if "duration_ms" in d:
            dur = _to_float(d.get("duration_ms", 0.0)) / 1000.0
        else:
            dur = _to_float(d.get("duration", 0.0))

        bytes_total = _to_int(d.get("bytes", d.get("total_bytes", 0)))
        pkts = _to_int(d.get("packets", d.get("total_packets", 0)))
        src_port = _to_int(d.get("src_port", 0))
        dst_port = _to_int(d.get("dst_port", 0))

        bytes_fwd = _to_int(d.get("bytes_fwd", bytes_total))
        bytes_bwd = _to_int(d.get("bytes_bwd", 0))
        packets_fwd = _to_int(d.get("packets_fwd", pkts))
        packets_bwd = _to_int(d.get("packets_bwd", 0))
        proto = d.get("proto", d.get("protocol", "TCP"))

        flow_id = d.get("flow_id") or _derive_flow_id(d, protocol_hint=str(proto))
        return FlowRecord(
            flow_id=flow_id,
            timestamp=raw.get("timestamp", ""),
            traffic_source="netflow",
            duration=dur,
            bytes_fwd=bytes_fwd,
            bytes_bwd=bytes_bwd,
            packets_fwd=packets_fwd,
            packets_bwd=packets_bwd,
            flow_bytes_per_sec=_safe_rate(bytes_fwd + bytes_bwd, dur),
            flow_pkts_per_sec=_safe_rate(packets_fwd + packets_bwd, dur),
            protocol=proto,
            src_port_bucket=_port_bucket(src_port),
            dst_port_bucket=_port_bucket(dst_port),
        )


class TLSParser:
    def parse(self, raw: dict) -> FlowRecord:
        d = raw.get("data", {})
        sni = d.get("sni", "")
        src_port = int(d.get("src_port", 0))
        dst_port = int(d.get("dst_port", 443))
        flow_id = d.get("flow_id") or _derive_flow_id(d, protocol_hint="TLS")
        return FlowRecord(
            flow_id=flow_id,
            timestamp=raw.get("timestamp", ""),
            traffic_source="tls",
            duration=float(d.get("duration", 0)),
            bytes_fwd=int(d.get("bytes_sent", 0)),
            bytes_bwd=int(d.get("bytes_recv", 0)),
            packets_fwd=int(d.get("records_sent", 0)),
            packets_bwd=int(d.get("records_recv", 0)),
            protocol="TLS",
            src_port_bucket=_port_bucket(src_port),
            dst_port_bucket=_port_bucket(dst_port),
            ja3_hash=d.get("ja3", ""),
            ja3s_hash=d.get("ja3s", ""),
            sni_entropy=_shannon_entropy(sni),
            cert_validity_days=int(d.get("cert_validity_days", -1)),
            tls_version=d.get("tls_version", ""),
        )


class AppLogParser:
    def parse(self, raw: dict) -> FlowRecord:
        d = raw.get("data", {})
        flow_id = d.get("flow_id") or _derive_flow_id(d, protocol_hint=d.get("protocol", "HTTP"))
        duration = _to_float(d.get("duration", 0.0))
        if duration == 0.0 and "latency_ms" in d:
            duration = _to_float(d.get("latency_ms", 0.0)) / 1000.0

        bytes_fwd = _to_int(d.get("request_bytes", d.get("bytes_fwd", 0)))
        bytes_bwd = _to_int(d.get("response_bytes", d.get("bytes_bwd", 0)))
        if bytes_fwd == 0 and bytes_bwd == 0:
            total_bytes = _to_int(d.get("total_bytes", 0))
            bytes_fwd = _to_int(d.get("bytes_fwd", total_bytes // 2))
            bytes_bwd = _to_int(d.get("bytes_bwd", total_bytes - bytes_fwd))

        packets_fwd = _to_int(d.get("packets_fwd", 0))
        packets_bwd = _to_int(d.get("packets_bwd", 0))
        if packets_fwd == 0 and packets_bwd == 0:
            total_packets = _to_int(d.get("total_packets", 0))
            packets_fwd = _to_int(d.get("packets_fwd", total_packets // 2))
            packets_bwd = _to_int(d.get("packets_bwd", total_packets - packets_fwd))

        dst_port = _to_int(d.get("port", d.get("dst_port", 80)))
        src_port = _to_int(d.get("src_port", 0))
        proto = d.get("protocol", "HTTP")

        return FlowRecord(
            flow_id=flow_id,
            timestamp=raw.get("timestamp", ""),
            traffic_source="app_log",
            duration=duration,
            bytes_fwd=bytes_fwd,
            bytes_bwd=bytes_bwd,
            packets_fwd=packets_fwd,
            packets_bwd=packets_bwd,
            flow_bytes_per_sec=_safe_rate(bytes_fwd + bytes_bwd, duration),
            flow_pkts_per_sec=_safe_rate(packets_fwd + packets_bwd, duration),
            src_port_bucket=_port_bucket(src_port),
            dst_port_bucket=_port_bucket(dst_port),
            protocol=proto,
        )


PARSERS = {
    "pcap": PcapParser(),
    "netflow": NetFlowParser(),
    "tls": TLSParser(),
    "app_log": AppLogParser(),
}


def parse_message(raw: dict) -> FlowRecord | None:
    source = raw.get("source", "")
    parser = PARSERS.get(source)
    if parser is None:
        return None
    return parser.parse(raw)
