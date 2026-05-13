from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import joblib

app = Flask(__name__)
CORS(app)

MODEL_FILE = "landmark_model.joblib"

# Load model once when server starts
try:
    model = joblib.load(MODEL_FILE)
    print(f"✅ Model loaded successfully: {MODEL_FILE}")
except Exception as e:
    model = None
    print(f"❌ Failed to load model: {e}")


@app.route("/", methods=["GET"])
def home():
    if model is None:
        return jsonify({
            "status": "error",
            "message": "Model not loaded"
        }), 500

    return jsonify({
        "status": "ASL landmark model API is running",
        "model_file": MODEL_FILE,
        "classes": list(model.classes_)
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({
                "error": "Model not loaded"
            }), 500

        data = request.get_json()

        if data is None:
            return jsonify({
                "error": "No JSON body received"
            }), 400

        if "landmarks" not in data:
            return jsonify({
                "error": "Missing 'landmarks' field"
            }), 400

        landmarks = data["landmarks"]

        if not isinstance(landmarks, list):
            return jsonify({
                "error": "'landmarks' must be a list"
            }), 400

        if len(landmarks) != 63:
            return jsonify({
                "error": f"Expected 63 landmark values, got {len(landmarks)}"
            }), 400

        landmarks = np.array(landmarks, dtype=float).reshape(1, -1)

        prediction = model.predict(landmarks)[0]

        confidence = 0.0
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(landmarks)[0]
            confidence = float(np.max(probabilities))

        return jsonify({
            "prediction": str(prediction),
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
