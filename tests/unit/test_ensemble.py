import pytest
from models.ensemble.arbitration import EnsembleArbitrator, Decision


@pytest.fixture
def arbitrator(monkeypatch):
    from unittest.mock import MagicMock
    mock_cfg = MagicMock()
    mock_cfg.models = {"ensemble": {"weights": {"lstm": 0.35, "xgboost": 0.40, "autoencoder": 0.25}}}
    mock_cfg.low_threshold = 0.35
    mock_cfg.high_threshold = 0.75
    monkeypatch.setattr("models.ensemble.arbitration.get_config", lambda: mock_cfg)
    return EnsembleArbitrator()


def test_allow_decision(arbitrator):
    result = arbitrator.arbitrate("flow1", lstm_score=0.1, xgb_score=0.1, ae_score=0.1)
    assert result.decision == Decision.ALLOW
    assert result.ensemble_score < 0.35


def test_block_decision(arbitrator):
    result = arbitrator.arbitrate("flow2", lstm_score=0.9, xgb_score=0.95, ae_score=0.85)
    assert result.decision == Decision.BLOCK
    assert result.ensemble_score >= 0.75


def test_queue_decision(arbitrator):
    result = arbitrator.arbitrate("flow3", lstm_score=0.5, xgb_score=0.55, ae_score=0.4)
    assert result.decision == Decision.QUEUE


def test_boundary_low(arbitrator):
    result = arbitrator.arbitrate("flow4", lstm_score=0.35, xgb_score=0.35, ae_score=0.35)
    assert result.decision == Decision.QUEUE


def test_weights_sum_to_one(arbitrator):
    total = sum(arbitrator._w.values())
    assert abs(total - 1.0) < 1e-6


def test_batch_length_matches(arbitrator):
    ids = ["f1", "f2", "f3"]
    import numpy as np
    scores = np.array([0.1, 0.5, 0.9])
    results = arbitrator.arbitrate_batch(ids, scores, scores, scores)
    assert len(results) == 3
    assert results[0].decision == Decision.ALLOW
    assert results[2].decision == Decision.BLOCK
