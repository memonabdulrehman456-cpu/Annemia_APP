"""AnemiaScan - Flask web app + REST API for anemia prediction."""
import json
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

from train_model import FEATURES, METRICS_PATH, MODEL_PATH, train

BASE = Path(__file__).parent
app = Flask(__name__, template_folder=str(BASE / "templates"))

# ---------------------------------------------------------------- model load
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        metrics = json.loads(METRICS_PATH.read_text())
        return model, metrics
    except Exception:  # missing file or sklearn version mismatch -> retrain
        return train(save=False)


MODEL, METRICS = load_model()

# Allowed input ranges (physiologically plausible values)
LIMITS = {
    "Hemoglobin": (3.0, 22.0),
    "MCH": (10.0, 40.0),
    "MCHC": (20.0, 40.0),
    "MCV": (50.0, 130.0),
}

# Typical adult reference ranges, used only for the "reading your values" chips
def reference(gender):
    return {
        "Hemoglobin": (13.5, 17.5) if gender == 1 else (12.0, 15.5),
        "MCH": (27.0, 33.0),
        "MCHC": (32.0, 36.0),
        "MCV": (80.0, 100.0),
    }


def parse_payload(data):
    """Validate request JSON. Returns (values, error)."""
    if not isinstance(data, dict):
        return None, "Send a JSON object."
    values = {}
    gender = data.get("gender", data.get("Gender"))
    if str(gender).lower() in ("1", "male", "m"):
        values["Gender"] = 1
    elif str(gender).lower() in ("0", "female", "f"):
        values["Gender"] = 0
    else:
        return None, "Please select gender (male or female)."

    for key, (lo, hi) in LIMITS.items():
        raw = data.get(key, data.get(key.lower()))
        try:
            num = float(raw)
        except (TypeError, ValueError):
            return None, f"{key} is required and must be a number."
        if not np.isfinite(num) or not lo <= num <= hi:
            return None, f"{key} must be between {lo:g} and {hi:g}."
        values[key] = num
    return values, None


# ------------------------------------------------------------------- routes
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok", model=METRICS.get("model"))


@app.get("/api/model-info")
def model_info():
    return jsonify(
        {
            "model": METRICS["model"],
            "accuracy": METRICS["accuracy"],
            "precision": METRICS["precision"],
            "recall": METRICS["recall"],
            "f1": METRICS["f1"],
            "samples_total": METRICS["samples_total"],
            "feature_importance": METRICS["feature_importance"],
            "feature_ranges": METRICS["feature_ranges"],
            "limits": LIMITS,
        }
    )


@app.post("/api/predict")
def predict():
    values, error = parse_payload(request.get_json(silent=True))
    if error:
        return jsonify(error=error), 400

    x = np.array([[values[f] for f in FEATURES]])
    proba = MODEL.predict_proba(x)[0]
    classes = list(MODEL.classes_)
    p_anemia = float(proba[classes.index(1)])
    anemic = p_anemia >= 0.5
    confidence = max(p_anemia, 1 - p_anemia)

    ref = reference(values["Gender"])
    flags = {}
    for key, (lo, hi) in ref.items():
        v = values[key]
        flags[key] = {
            "value": v,
            "status": "low" if v < lo else "high" if v > hi else "normal",
            "range": [lo, hi],
        }

    # Heads-up when the input is outside what the model was trained on
    outside = [
        k
        for k in LIMITS
        if not METRICS["feature_ranges"][k][0] <= values[k] <= METRICS["feature_ranges"][k][1]
    ]

    return jsonify(
        prediction=int(anemic),
        label="Anemia detected" if anemic else "No anemia detected",
        probability_anemia=round(p_anemia, 4),
        probability_healthy=round(1 - p_anemia, 4),
        confidence=round(confidence, 4),
        confidence_level="high" if confidence >= 0.85 else "moderate" if confidence >= 0.65 else "low",
        flags=flags,
        outside_training_range=outside,
    )


@app.errorhandler(404)
def not_found(_):
    if request.path.startswith("/api/"):
        return jsonify(error="Not found"), 404
    return render_template("index.html"), 404


@app.errorhandler(500)
def server_error(_):
    return jsonify(error="Something went wrong on the server."), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
