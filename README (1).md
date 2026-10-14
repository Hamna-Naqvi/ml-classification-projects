# Banknote Authenticity Checker

Classifies a banknote as genuine or forged from four statistics of its wavelet-transformed image (variance, skewness, curtosis, entropy).

| | |
|---|---|
| Model | Logistic Regression (with `StandardScaler`) |
| Split | 80/20, stratified, `random_state=42` |
| Test accuracy | **97.1%** |
| Data | 1372 rows, 5 columns (4 features + class) |



## Quick start

From the repository root:

```bash
pip install -r requirements.txt
cd banknote-authentication

# 1. put BankNote_Authentication.csv in data/  (see data/README.md)
python train.py          # trains and saves model/model.joblib

# 2. run a web app (pick one)
python app_gradio.py     # http://127.0.0.1:7860
python app_flask.py      # http://127.0.0.1:5000
```

## Contents

| File | Purpose |
|---|---|
| `notebooks/banknote_authentication.ipynb` | Exploration, training, evaluation and the two demo apps |
| `train.py` | Reproducible training script, saves the model and metrics |
| `common.py` | Shared settings and model loading |
| `app_gradio.py` | Gradio interface |
| `app_flask.py` | Flask interface |
| `data/` | Place the CSV here (not tracked) |
