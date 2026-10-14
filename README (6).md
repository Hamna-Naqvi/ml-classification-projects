# Heart Failure Survival Predictor

Predicts whether a heart failure patient is likely to die during the follow-up period, from clinical records.

| | |
|---|---|
| Model | Logistic Regression (with `StandardScaler`) |
| Split | 80/20, stratified, `random_state=42` |
| Test accuracy | **81.7%** |
| Data | 299 rows, 13 columns, no missing values |

**Note:** recall for the death class is only 0.58 (8 of 19 deaths missed). For a medical use case that matters more than overall accuracy. This is an educational project, **not** a medical tool.

## Quick start

From the repository root:

```bash
pip install -r requirements.txt
cd heart-failure-prediction

# 1. put heart_failure_clinical_records_dataset.csv in data/  (see data/README.md)
python train.py          # trains and saves model/model.joblib

# 2. run a web app (pick one)
python app_gradio.py     # http://127.0.0.1:7860
python app_flask.py      # http://127.0.0.1:5000
```

## Contents

| File | Purpose |
|---|---|
| `notebooks/heart_failure_prediction.ipynb` | Exploration, training, evaluation and the two demo apps |
| `train.py` | Reproducible training script, saves the model and metrics |
| `common.py` | Shared settings and model loading |
| `app_gradio.py` | Gradio interface |
| `app_flask.py` | Flask interface |
| `data/` | Place the CSV here (not tracked) |
