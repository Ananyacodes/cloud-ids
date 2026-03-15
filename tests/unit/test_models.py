import pytest
import torch
import numpy as np
from models.lstm.model import LSTMClassifier
from models.autoencoder.model import Autoencoder


def test_lstm_forward_shape():
    model = LSTMClassifier(input_size=21, hidden_size=64, num_layers=2, dropout=0.0)
    model.eval()
    x = torch.randn(8, 32, 21)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (8,)


def test_lstm_predict_proba_range():
    model = LSTMClassifier(input_size=21, hidden_size=64, num_layers=2, dropout=0.0)
    model.eval()
    x = torch.randn(4, 32, 21)
    probs = model.predict_proba(x)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_autoencoder_reconstruction_shape():
    model = Autoencoder(input_dim=21, hidden_dims=[32, 16], latent_dim=8, dropout=0.0)
    model.eval()
    x = torch.randn(16, 21)
    with torch.no_grad():
        recon = model(x)
    assert recon.shape == x.shape


def test_autoencoder_error_positive():
    model = Autoencoder(input_dim=21, hidden_dims=[32, 16], latent_dim=8, dropout=0.0)
    model.eval()
    x = torch.randn(4, 21)
    errors = model.reconstruction_error(x)
    assert (errors >= 0).all()
    assert errors.shape == (4,)


def test_autoencoder_threshold_fitting():
    model = Autoencoder(input_dim=21, hidden_dims=[32, 16], latent_dim=8, dropout=0.0)
    model.eval()
    benign = np.random.randn(500, 21).astype(np.float32)
    threshold = model.fit_threshold(benign, percentile=95)
    assert threshold > 0
    assert model.threshold == threshold


def test_autoencoder_proba_range():
    model = Autoencoder(input_dim=21, hidden_dims=[32, 16], latent_dim=8, dropout=0.0)
    model.eval()
    model.threshold = 0.05
    x = torch.randn(8, 21)
    probs = model.predict_proba(x)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0
