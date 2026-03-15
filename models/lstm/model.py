from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 21,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_size * 2
        self.attention = nn.Linear(lstm_out_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    # Runs the sequence through the bidirectional LSTM, then uses a learned
    # attention layer to weight each timestep before summing into a context
    # vector that the classification head maps to a scalar threat logit.
    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(out), dim=1)
        context = (attn_weights * out).sum(dim=1)
        context = self.dropout(context)
        return self.classifier(context).squeeze(-1)

    def predict_proba(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))
