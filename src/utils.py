from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def load_config(root: Path) -> dict:
    with open(root / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def service_level_to_z(service_level: float) -> float:
    """Approximate normal z-score values for safety-stock calculations."""
    mapping = {
        0.80: 0.84,
        0.85: 1.04,
        0.90: 1.28,
        0.92: 1.41,
        0.95: 1.65,
        0.97: 1.88,
        0.98: 2.05,
        0.99: 2.33,
    }
    closest = min(mapping, key=lambda x: abs(float(x) - float(service_level)))
    return mapping[closest]


def safe_mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.where(np.abs(y_true) < 1e-6, 1.0, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100)


def ensure_dirs(root: Path) -> None:
    for folder in ["data/raw", "data/processed", "data/powerbi_outputs", "models"]:
        (root / folder).mkdir(parents=True, exist_ok=True)
