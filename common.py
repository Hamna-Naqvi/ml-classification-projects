"""Shared settings and model loading for the web apps."""
import json
from pathlib import Path

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).parent / "model"

DEFAULTS = {'variance': 0.0, 'skewness': 0.0, 'curtosis': 0.0, 'entropy': 0.0}
POS, NEG = "Forged banknote", "Genuine banknote"
TITLE = "Banknote Authenticity Checker"
NOTE = "Features come from wavelet-transformed banknote images."


def load():
    if not (MODEL_DIR / "model.joblib").exists():
        raise SystemExit("No trained model found. Run `python train.py` first.")
    model = joblib.load(MODEL_DIR / "model.joblib")
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    return model, meta["features"], meta["accuracy"]


def predict_proba(model, features, values):
    """Probability of the positive class for one row of values."""
    row = pd.DataFrame([list(values)], columns=features)
    return float(model.predict_proba(row)[0][1])
