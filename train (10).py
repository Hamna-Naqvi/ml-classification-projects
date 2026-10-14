"""Train the Heart Failure Survival Predictor model and save it to model/.

Usage:
    python train.py                       # uses data/heart_failure_clinical_records_dataset.csv
    python train.py --data path/to/file.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
TARGET = "DEATH_EVENT"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=HERE / "data" / "heart_failure_clinical_records_dataset.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    print("Shape:", df.shape)
    print(df[TARGET].value_counts())

    X = df.drop(columns=TARGET)
    y = df[TARGET]
    features = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy:", round(acc, 4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    out = HERE / "model"
    out.mkdir(exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    (out / "meta.json").write_text(json.dumps({"features": features, "accuracy": acc}, indent=2))
    print("Saved model to", out)


if __name__ == "__main__":
    main()
