"""
Titanic Survival Predictor — Flask Backend (app.py)

Production-ready Flask REST API serving a pre-trained Scikit-learn Pipeline
(StandardScaler → Logistic Regression) for Titanic passenger survival prediction.

Configured for local development and cloud production deployment on Render.

Author: Bunny
"""

import os
import traceback

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ---------------------------------------------------------------------------
# App Initialization (Standard Flask layout: templates/ and static/)
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for API requests

# ---------------------------------------------------------------------------
# Load Trained Model (once at startup)
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "titanic_model.pkl")

try:
    loaded_model = joblib.load(MODEL_PATH)
    print(f"[✓] Model loaded successfully from: {MODEL_PATH}")
    print(f"    Pipeline: {loaded_model}")
    if hasattr(loaded_model, "feature_names_in_"):
        print(f"    Expected features: {list(loaded_model.feature_names_in_)}")
except Exception as e:
    print(f"[✗] Failed to load model: {e}")
    loaded_model = None

# ---------------------------------------------------------------------------
# Feature Column Order (must match model training exactly)
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "Embarked_Q",
    "Embarked_S",
]

# ---------------------------------------------------------------------------
# Routes — Frontend Rendering
# ---------------------------------------------------------------------------


@app.route("/")
def serve_index():
    """Serve the main frontend HTML template via render_template."""
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Routes — Prediction REST API Endpoint
# ---------------------------------------------------------------------------


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict

    Accepts JSON with passenger parameters, computes FamilySize and
    one-hot encoded Embarked columns, runs inference through the
    loaded Scikit-learn Pipeline, and returns prediction with probability.

    Expected JSON keys:
        Pclass / pclass   (int)     — 1, 2, or 3
        Sex / sex        (str/int) — "female"/"male" or 1/0 (female=1, male=0)
        Age / age        (float)   — passenger age
        Fare / fare      (float)   — ticket fare
        SibSp / sibsp    (int)     — siblings / spouses aboard
        Parch / parch    (int)     — parents / children aboard
        Embarked / embarked (str)  — "S", "Q", or "C"

    Returns JSON:
        prediction       (str)   — "Survived" or "Did Not Survive"
        prediction_value (int)   — 1 or 0
        probability      (float) — survival probability as percentage
    """

    # --- Guard: Model availability ---
    if loaded_model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 503

    # --- Guard: Content-Type check ---
    if not request.is_json:
        return jsonify({"error": "Request must be JSON (Content-Type: application/json)."}), 400

    data = request.get_json()

    # ------------------------------------------------------------------
    # 1. Extract & Validate Input Features
    # ------------------------------------------------------------------

    try:
        # Accept both capitalised and lowercase parameter keys
        pclass = int(data.get("Pclass") if data.get("Pclass") is not None else data.get("pclass"))
        age = float(data.get("Age") if data.get("Age") is not None else data.get("age"))
        fare = float(data.get("Fare") if data.get("Fare") is not None else data.get("fare"))
        sibsp = int(data.get("SibSp") if data.get("SibSp") is not None else data.get("sibsp", 0))
        parch = int(data.get("Parch") if data.get("Parch") is not None else data.get("parch", 0))
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid or missing numeric field: {e}"}), 400

    # --- Sex Feature Encoding (Trained model encoding: female=1, male=0) ---
    raw_sex = data.get("Sex") if data.get("Sex") is not None else data.get("sex")
    if raw_sex is None:
        return jsonify({"error": "Missing required field: sex"}), 400

    if isinstance(raw_sex, str):
        sex = 1 if raw_sex.strip().lower() == "female" else 0
    else:
        sex = int(raw_sex)

    # --- Embarked Feature One-Hot Encoding (Embarked_Q, Embarked_S) ---
    raw_embarked = str(data.get("Embarked") if data.get("Embarked") is not None else data.get("embarked", "S")).strip().upper()

    if raw_embarked not in ("S", "Q", "C"):
        return jsonify({"error": f"Invalid embarked value: '{raw_embarked}'. Must be S, Q, or C."}), 400

    embarked_q = 1 if raw_embarked == "Q" else 0
    embarked_s = 1 if raw_embarked == "S" else 0
    # Note: Cherbourg ("C") → Embarked_Q=0, Embarked_S=0 (baseline category)

    # --- Feature Range Validation ---
    if pclass not in (1, 2, 3):
        return jsonify({"error": "Pclass must be 1, 2, or 3."}), 400
    if not (0 <= age <= 100):
        return jsonify({"error": "Age must be between 0 and 100."}), 400
    if fare < 0:
        return jsonify({"error": "Fare must be non-negative."}), 400
    if sibsp < 0 or parch < 0:
        return jsonify({"error": "SibSp and Parch must be non-negative."}), 400

    # ------------------------------------------------------------------
    # 2. Compute FamilySize Automatically
    # ------------------------------------------------------------------

    family_size = sibsp + parch + 1

    # ------------------------------------------------------------------
    # 3. Construct Feature DataFrame (exact feature order from training)
    # ------------------------------------------------------------------

    features_df = pd.DataFrame(
        [{
            "Pclass": pclass,
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "FamilySize": family_size,
            "Embarked_Q": embarked_q,
            "Embarked_S": embarked_s,
        }],
        columns=FEATURE_COLUMNS,
    )

    # ------------------------------------------------------------------
    # 4. Perform Model Inference
    # ------------------------------------------------------------------

    try:
        prediction_value = int(loaded_model.predict(features_df)[0])
        probabilities = loaded_model.predict_proba(features_df)[0]

        # Class 1 probability represents survival probability
        survival_prob = float(probabilities[1])
        probability_pct = round(survival_prob * 100, 2)

        prediction_label = "Survived" if prediction_value == 1 else "Did Not Survive"

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Model prediction failed: {str(e)}"}), 500

    # ------------------------------------------------------------------
    # 5. Return Structured Response
    # ------------------------------------------------------------------

    return jsonify({
        "prediction": prediction_label,
        "prediction_value": prediction_value,
        "probability": probability_pct,
    })


# ---------------------------------------------------------------------------
# Global Error Handlers
# ---------------------------------------------------------------------------


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "HTTP method not allowed."}), 405


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error."}), 500


# ---------------------------------------------------------------------------
# Server Entrypoint (Production & Development Server)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Retrieve port from PORT environment variable (Render sets this dynamically)
    port = int(os.environ.get("PORT", 5001))
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() in ["true", "1"]
    
    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug_mode
    )
