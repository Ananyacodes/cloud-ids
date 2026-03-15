import pytest
from ingestion.parsers import AppLogParser, PcapParser, NetFlowParser, TLSParser, parse_message


@pytest.fixture
def pcap_msg():
    return {
        "source": "pcap",
        "timestamp": "2024-01-15T10:00:00Z",
        "data": {
            "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2",
            "src_port": 54321, "dst_port": 443,
            "duration": 1.5,
            "total_bytes": 4096, "total_packets": 20,
            "bytes_fwd": 1024, "bytes_bwd": 3072,
            "packets_fwd": 8, "packets_bwd": 12,
            "pkt_size_mean": 204.8, "pkt_size_std": 50.0,
            "iat_mean": 0.075, "iat_std": 0.01,
            "syn_flag_cnt": 1, "ack_flag_cnt": 18,
            "fin_flag_cnt": 1, "rst_flag_cnt": 0, "psh_flag_cnt": 5,
            "protocol": "TCP",
        }
    }


def test_pcap_parser_basic(pcap_msg):
    record = PcapParser().parse(pcap_msg)
    assert record.traffic_source == "pcap"
    assert record.duration == 1.5
    assert record.bytes_fwd == 1024
    assert record.syn_flag_cnt == 1
    assert record.src_port_bucket == "ephemeral"
    assert record.dst_port_bucket == "well_known"


def test_pcap_parser_rates(pcap_msg):
    record = PcapParser().parse(pcap_msg)
    assert abs(record.flow_bytes_per_sec - 4096 / 1.5) < 0.1
    assert abs(record.flow_pkts_per_sec - 20 / 1.5) < 0.1


def test_pcap_zero_duration():
    msg = {"source": "pcap", "timestamp": "", "data": {"duration": 0, "total_bytes": 100, "total_packets": 5}}
    record = PcapParser().parse(msg)
    assert record.flow_bytes_per_sec == 0.0


def test_tls_sni_entropy():
    msg = {
        "source": "tls", "timestamp": "",
        "data": {"sni": "evil-c2-server.biz", "ja3": "abc123", "tls_version": "TLSv1.3",
                 "bytes_sent": 512, "bytes_recv": 2048,
                 "src_port": 55000, "dst_port": 443}
    }
    record = TLSParser().parse(msg)
    assert record.sni_entropy > 0
    assert record.tls_version == "TLSv1.3"


def test_parse_message_unknown_source():
    assert parse_message({"source": "unknown"}) is None


def test_parse_message_dispatches_correctly(pcap_msg):
    record = parse_message(pcap_msg)
    assert record is not None
    assert record.traffic_source == "pcap"


def test_netflow_parser_accepts_common_netflow_keys():
    msg = {
        "source": "netflow",
        "timestamp": "2024-01-15T10:01:00Z",
        "data": {
            "src_ip": "10.0.1.1",
            "dst_ip": "10.0.0.2",
            "src_port": 56466,
            "dst_port": 443,
            "duration": 1.82,
            "total_bytes": 7217,
            "total_packets": 178,
            "protocol": "TCP",
        },
    }
    record = NetFlowParser().parse(msg)
    assert record.duration == pytest.approx(1.82)
    assert record.bytes_fwd == 7217
    assert record.packets_fwd == 178
    assert record.protocol == "TCP"
    assert len(record.flow_id) == 16


def test_applog_flow_id_is_stable_for_same_network_tuple():
    base = {
        "src_ip": "10.0.2.1",
        "dst_ip": "10.0.0.2",
        "src_port": 58267,
        "dst_port": 443,
        "protocol": "TCP",
        "duration": 1.25,
        "total_bytes": 900,
        "total_packets": 9,
    }
    reordered = {
        "total_packets": 9,
        "protocol": "TCP",
        "dst_port": 443,
        "src_port": 58267,
        "src_ip": "10.0.2.1",
        "duration": 1.25,
        "dst_ip": "10.0.0.2",
        "total_bytes": 900,
    }

    parser = AppLogParser()
    r1 = parser.parse({"source": "app_log", "timestamp": "", "data": base})
    r2 = parser.parse({"source": "app_log", "timestamp": "", "data": reordered})

    assert r1.flow_id == r2.flow_id
