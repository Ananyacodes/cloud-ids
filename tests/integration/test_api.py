from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_predictor():
    from models.ensemble.arbitration import EnsembleResult, Decision
    mock = MagicMock()
    mock.predict_one.return_value = EnsembleResult(
        flow_id="test-flow",
        lstm_score=0.8,
        xgb_score=0.85,
        ae_score=0.7,
        ensemble_score=0.8,
        decision=Decision.BLOCK,
    )
    mock.predict_batch.return_value = [
        EnsembleResult(
            flow_id=f"flow-{i}",
            lstm_score=0.1,
            xgb_score=0.1,
            ae_score=0.1,
            ensemble_score=0.1,
            decision=Decision.ALLOW,
        )
        for i in range(3)
    ]
    return mock


@pytest.fixture
def mock_router():
    mock = MagicMock()
    mock.route.return_value = True
    return mock


@pytest.fixture
def client(mock_predictor, mock_router):
    with patch("inference.api.app._predictor", mock_predictor), \
         patch("inference.api.app._router", mock_router):
        from inference.api.app import app
        with TestClient(app) as c:
            yield c


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_readyz(client):
    resp = client.get("/readyz")
    assert resp.status_code == 200


def test_predict_returns_decision(client):
    payload = {
        "flow_id": "test-flow",
        "timestamp": "2024-01-15T10:00:00Z",
        "traffic_source": "netflow",
        "duration": 5.0,
        "bytes_fwd": 50000,
        "packets_fwd": 100,
        "flow_bytes_per_sec": 10000.0,
        "flow_pkts_per_sec": 20.0,
        "protocol": "TCP",
        "src_port_bucket": "ephemeral",
        "dst_port_bucket": "well_known",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["flow_id"] == "test-flow"
    assert data["decision"] == "block"
    assert 0.0 <= data["ensemble_score"] <= 1.0


def test_predict_batch(client):
    flows = [
        {
            "flow_id": f"flow-{i}",
            "traffic_source": "netflow",
            "protocol": "TCP",
            "src_port_bucket": "ephemeral",
            "dst_port_bucket": "well_known",
        }
        for i in range(3)
    ]
    resp = client.post("/predict/batch", json={"flows": flows})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 3
    assert "latency_ms" in data


def test_predict_missing_flow_id(client):
    resp = client.post("/predict", json={"traffic_source": "netflow"})
    assert resp.status_code == 422  # validation error
