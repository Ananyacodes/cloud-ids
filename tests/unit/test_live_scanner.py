from __future__ import annotations

import pytest

from ingestion.live_scanner import LivePacketScanner


def test_live_scanner_aggregates_bidirectional_tcp_flow():
    scapy = __import__("scapy.all", fromlist=["Ether", "IP", "TCP", "Raw"])
    Ether, IP, TCP, Raw = scapy.Ether, scapy.IP, scapy.TCP, scapy.Raw

    emitted: list[dict] = []
    scanner = LivePacketScanner(on_flow=emitted.append, flow_idle_timeout=0.0)

    p1 = Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=50000, dport=443, flags="S") / Raw(load="a" * 40)
    p2 = Ether() / IP(src="10.0.0.2", dst="10.0.0.1") / TCP(sport=443, dport=50000, flags="SA") / Raw(load="b" * 20)

    p1.time = 1000.0
    p2.time = 1000.2

    scanner._process_packet(p1)
    scanner._process_packet(p2)
    emitted_count = scanner._flush_expired(force=True)

    assert emitted_count == 1
    assert len(emitted) == 1

    msg = emitted[0]
    assert msg["source"] == "pcap"
    assert msg["data"]["protocol"] == "TCP"
    assert msg["data"]["packets_fwd"] == 1
    assert msg["data"]["packets_bwd"] == 1
    assert msg["data"]["syn_flag_cnt"] >= 2
    assert msg["data"]["ack_flag_cnt"] >= 1
    assert msg["data"]["duration"] == pytest.approx(0.2)
