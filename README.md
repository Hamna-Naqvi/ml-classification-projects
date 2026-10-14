# ML Classification Projects

Two small end-to-end machine learning projects built with scikit-learn. Each one trains a Logistic Regression classifier on a public Kaggle dataset and serves it through both a Gradio and a Flask web interface.

| Project | Task | Test accuracy |
|---|---|---|
| [heart-failure-prediction](heart-failure-prediction/) | Predict patient death event from clinical records | 81.7% |
| [banknote-authentication](banknote-authentication/) | Detect forged vs genuine banknotes | 97.1% |

## Tech stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Gradio, Flask, Jupyter

## Repository structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── heart-failure-prediction/
│   ├── notebooks/        # Jupyter notebook
│   ├── data/             # put the CSV here (not tracked)
│   ├── train.py          # train and save model
│   ├── common.py
│   ├── app_gradio.py
│   └── app_flask.py
└── banknote-authentication/
    └── (same layout)
```

## Getting started

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then follow the README inside each project folder (download the dataset, run `train.py`, launch an app).

## Pipeline

1. Load the CSV and inspect shape, missing values and class balance
2. Stratified 80/20 train/test split
3. `StandardScaler` + `LogisticRegression` in a single pipeline
4. Evaluate with accuracy, confusion matrix and classification report
5. Serve predictions through a web form

## Disclaimer

These are learning projects. The heart failure model in particular is **not** a medical device and must not be used for clinical decisions.

## License

MIT, see [LICENSE](LICENSE).
