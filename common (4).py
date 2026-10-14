"""Shared settings and model loading for the web apps."""
import json
from pathlib import Path

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).parent / "model"

DEFAULTS = {'age':60,'anaemia':0,'creatinine_phosphokinase':250,'diabetes':0,
            'ejection_fraction':38,'high_blood_pressure':0,'platelets':263000,
            'serum_creatinine':1.1,'serum_sodium':137,'sex':1,'smoking':0,'time':100}
POS, NEG = "Death event likely", "Survival likely"
TITLE = "Heart Failure Survival Predictor"
NOTE = "Binary columns (anaemia, diabetes, high_blood_pressure, sex, smoking) take 0 or 1."


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
