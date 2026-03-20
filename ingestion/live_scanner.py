from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Callable

from scapy.all import IP, TCP, UDP, sniff

logger = logging.getLogger(__name__)


@dataclass
class FlowAccumulator:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    start_ts: float
    last_ts: float
    bytes_fwd: int = 0
    bytes_bwd: int = 0
    packets_fwd: int = 0
    packets_bwd: int = 0
    pkt_sizes: list[int] = field(default_factory=list)
    iats: list[float] = field(default_factory=list)
    fin_flag_cnt: int = 0
    syn_flag_cnt: int = 0
    rst_flag_cnt: int = 0
    psh_flag_cnt: int = 0
    ack_flag_cnt: int = 0

    def update(
        self,
        *,
        ts: float,
        size: int,
        src_ip: str,
        src_port: int,
        dst_ip: str,
        dst_port: int,
        tcp_flags: int | None,
    ) -> None:
        is_forward = (
            src_ip == self.src_ip
            and dst_ip == self.dst_ip
            and src_port == self.src_port
            and dst_port == self.dst_port
        )

        if is_forward:
            self.bytes_fwd += size
            self.packets_fwd += 1
        else:
            self.bytes_bwd += size
            self.packets_bwd += 1

        if ts > self.last_ts:
            self.iats.append(ts - self.last_ts)

        self.last_ts = max(self.last_ts, ts)
        self.pkt_sizes.append(size)

        if tcp_flags is None:
            return

        if tcp_flags & 0x01:
            self.fin_flag_cnt += 1
        if tcp_flags & 0x02:
            self.syn_flag_cnt += 1
        if tcp_flags & 0x04:
            self.rst_flag_cnt += 1
        if tcp_flags & 0x08:
            self.psh_flag_cnt += 1
        if tcp_flags & 0x10:
            self.ack_flag_cnt += 1

    def to_message(self) -> dict:
        import numpy as np

        total_bytes = self.bytes_fwd + self.bytes_bwd
        total_packets = self.packets_fwd + self.packets_bwd
        duration = max(0.0, self.last_ts - self.start_ts)

        pkt_size_mean = float(np.mean(self.pkt_sizes)) if self.pkt_sizes else 0.0
        pkt_size_std = float(np.std(self.pkt_sizes)) if self.pkt_sizes else 0.0
        iat_mean = float(np.mean(self.iats)) if self.iats else 0.0
        iat_std = float(np.std(self.iats)) if self.iats else 0.0

        ts_iso = (
            dt.datetime.fromtimestamp(self.last_ts, dt.UTC)
            .isoformat()
            .replace("+00:00", "Z")
        )

        return {
            "source": "pcap",
            "timestamp": ts_iso,
            "data": {
                "src_ip": self.src_ip,
                "dst_ip": self.dst_ip,
                "src_port": self.src_port,
                "dst_port": self.dst_port,
                "protocol": self.protocol,
                "duration": duration,
                "total_bytes": total_bytes,
                "total_packets": total_packets,
                "bytes_fwd": self.bytes_fwd,
                "bytes_bwd": self.bytes_bwd,
                "packets_fwd": self.packets_fwd,
                "packets_bwd": self.packets_bwd,
                "pkt_size_mean": pkt_size_mean,
                "pkt_size_std": pkt_size_std,
                "iat_mean": iat_mean,
                "iat_std": iat_std,
                "fin_flag_cnt": self.fin_flag_cnt,
                "syn_flag_cnt": self.syn_flag_cnt,
                "rst_flag_cnt": self.rst_flag_cnt,
                "psh_flag_cnt": self.psh_flag_cnt,
                "ack_flag_cnt": self.ack_flag_cnt,
            },
        }


class LivePacketScanner:
    def __init__(
        self,
        on_flow: Callable[[dict], None],
        interface: str | None = None,
        bpf_filter: str = "ip",
        flow_idle_timeout: float = 5.0,
    ) -> None:
        self._on_flow = on_flow
        self._interface = interface
        self._bpf_filter = bpf_filter
        self._flow_idle_timeout = flow_idle_timeout
        self._flows: dict[tuple, FlowAccumulator] = {}
        self._running = False

    @staticmethod
    def _build_flow_keys(pkt) -> tuple[tuple, tuple] | None:
        if not pkt.haslayer(IP):
            return None

        ip = pkt[IP]
        src_ip = str(ip.src)
        dst_ip = str(ip.dst)

        if pkt.haslayer(TCP):
            l4 = pkt[TCP]
            src_port = int(l4.sport)
            dst_port = int(l4.dport)
            protocol = "TCP"
        elif pkt.haslayer(UDP):
            l4 = pkt[UDP]
            src_port = int(l4.sport)
            dst_port = int(l4.dport)
            protocol = "UDP"
        else:
            src_port = 0
            dst_port = 0
            protocol = str(ip.proto)

        canonical = tuple(sorted(((src_ip, src_port), (dst_ip, dst_port))))
        key = (canonical, protocol)
        direction = (src_ip, src_port, dst_ip, dst_port)
        return key, direction

    def _process_packet(self, pkt) -> None:
        built = self._build_flow_keys(pkt)
        if built is None:
            return

        key, direction = built
        src_ip, src_port, dst_ip, dst_port = direction
        ts = float(getattr(pkt, "time", dt.datetime.now(dt.UTC).timestamp()))
        size = int(len(pkt))
        tcp_flags = int(pkt[TCP].flags) if pkt.haslayer(TCP) else None

        acc = self._flows.get(key)
        if acc is None:
            protocol = key[1]
            acc = FlowAccumulator(
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                protocol=protocol,
                start_ts=ts,
                last_ts=ts,
            )
            self._flows[key] = acc

        acc.update(
            ts=ts,
            size=size,
            src_ip=src_ip,
            src_port=src_port,
            dst_ip=dst_ip,
            dst_port=dst_port,
            tcp_flags=tcp_flags,
        )

    def _flush_expired(self, *, force: bool = False) -> int:
        now = dt.datetime.now(dt.UTC).timestamp()
        emitted = 0
        to_delete: list[tuple] = []

        for key, acc in self._flows.items():
            idle_for = now - acc.last_ts
            if force or idle_for >= self._flow_idle_timeout:
                self._on_flow(acc.to_message())
                to_delete.append(key)
                emitted += 1

        for key in to_delete:
            del self._flows[key]

        return emitted

    def start(self) -> None:
        self._running = True
        logger.info(
            "Starting live scanner interface=%s filter=%s idle_timeout=%.2fs",
            self._interface or "default",
            self._bpf_filter,
            self._flow_idle_timeout,
        )

        while self._running:
            sniff(
                iface=self._interface,
                filter=self._bpf_filter,
                prn=self._process_packet,
                store=False,
                timeout=1,
            )
            self._flush_expired()

    def stop(self) -> None:
        self._running = False
        emitted = self._flush_expired(force=True)
        logger.info("Stopped live scanner. Emitted remaining flows=%d", emitted)
