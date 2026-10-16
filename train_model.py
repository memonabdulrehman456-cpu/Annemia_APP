"""Train the anemia classifier.

Run:  python train_model.py
Creates: model.joblib and metrics.json
(app.py also calls train() automatically if model.joblib is missing or
cannot be loaded, so deployment never breaks because of a version mismatch.)
"""
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "anemia.csv"
MODEL_PATH = BASE / "model.joblib"
METRICS_PATH = BASE / "metrics.json"

# Gender: 0 = Female, 1 = Male (confirmed from the data: anemia cut-off is
# ~12 g/dL for group 0 and ~13.5 g/dL for group 1).
FEATURES = ["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"]
TARGET = "Result"  # 1 = anemic, 0 = not anemic


def load_data(path=DATA_PATH):
    rows, seen = [], set()
    with open(path, newline="") as f:
        for d in csv.DictReader(f):
            try:
                vals = tuple(float(d[k]) for k in FEATURES + [TARGET])
            except (KeyError, TypeError, ValueError):
                continue  # skip incomplete rows
            if vals in seen:  # drop exact duplicates (avoids train/test leakage)
                continue
            seen.add(vals)
            rows.append(vals)
    arr = np.array(rows)
    return arr[:, :-1], arr[:, -1].astype(int)


def train(save=True):
    X, y = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    candidates = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=6, min_samples_leaf=2, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    comparison = {
        name: round(float(cross_val_score(m, X_tr, y_tr, cv=5).mean()), 4)
        for name, m in candidates.items()
    }

    # Final model: Random Forest (smooth probabilities, robust, small file)
    model = candidates["Random Forest"]
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    cm = confusion_matrix(y_te, pred)

    metrics = {
        "model": "Random Forest",
        "samples_total": int(len(y)),
        "samples_train": int(len(y_tr)),
        "samples_test": int(len(y_te)),
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "precision": round(float(precision_score(y_te, pred)), 4),
        "recall": round(float(recall_score(y_te, pred)), 4),
        "f1": round(float(f1_score(y_te, pred)), 4),
        "confusion_matrix": cm.tolist(),
        "cv_accuracy_comparison": comparison,
        "feature_importance": {
            k: round(float(v), 4) for k, v in zip(FEATURES, model.feature_importances_)
        },
        "feature_ranges": {
            k: [float(X[:, i].min()), float(X[:, i].max())] for i, k in enumerate(FEATURES)
        },
    }

    # Refit on all (de-duplicated) data for the deployed model
    model.fit(X, y)
    if save:
        joblib.dump(model, MODEL_PATH, compress=3)
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return model, metrics


if __name__ == "__main__":
    _, m = train()
    print(json.dumps(m, indent=2))
