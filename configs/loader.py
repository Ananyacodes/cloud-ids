from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class Config:
    gcp: dict
    ingestion: dict
    features: dict
    models: dict
    triage: dict
    training: dict
    retraining: dict
    serving: dict

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        d = _load_yaml(str(path))
        return cls(**d)

    @property
    def low_threshold(self) -> float:
        return float(self.triage["low_threshold"])

    @property
    def high_threshold(self) -> float:
        return float(self.triage["high_threshold"])

    @property
    def sequence_length(self) -> int:
        return int(self.features["sequence_length"])

    @property
    def numerical_cols(self) -> list[str]:
        return self.features["numerical_cols"]

    @property
    def categorical_cols(self) -> list[str]:
        return self.features["categorical_cols"]


_CONFIG: Config | None = None


def get_config(path: str | None = None) -> Config:
    global _CONFIG
    if _CONFIG is None:
        p = path or os.environ.get("IDS_CONFIG", "configs/config.yaml")
        _CONFIG = Config.from_yaml(p)
    return _CONFIG
