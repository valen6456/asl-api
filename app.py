from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import joblib

app = Flask(__name__)
CORS(app)

# =========================
# Model files
# =========================
ALPHABET_MODEL_FILE = "landmark_model.joblib"
GESTURE_MODEL_FILE = "gesture_model.joblib"

alphabet_model = None
gesture_model = None


# =========================
# Thai text mapping
# =========================
GESTURE_TEXT_TH = {
    "HELLO": "สวัสดี",
    "THANK_YOU": "ขอบคุณ",
    "SORRY": "ขอโทษ",
    "GOODBYE": "ลาก่อน",
    "YES": "ใช่",
    "NO": "ไม่ใช่",
    "OK": "โอเค",
    "EAT": "กินข้าว",
    "DRINK": "ดื่มน้ำ",
    "SLEEPY": "ง่วงแล้ว",
}


# =========================
# Load models
# =========================
def load_models():
    global alphabet_model, gesture_model

    try:
        alphabet_model = joblib.load(ALPHABET_MODEL_FILE)
        print(f"✅ Alphabet model loaded: {ALPHABET_MODEL_FILE}")
    except Exception as e:
        alphabet_model = None
        print(f"❌ Failed to load alphabet model: {e}")

    try:
        gesture_model = joblib.load(GESTURE_MODEL_FILE)
        print(f"✅ Gesture model loaded: {GESTURE_MODEL_FILE}")
    except Exception as e:
        gesture_model = None
        print(f"❌ Failed to load gesture model: {e}")


load_models()


# =========================
# Home route
# =========================
@app.route("/", methods=["GET"])
def home():
    alphabet_classes = []
    gesture_classes = []

    if alphabet_model is not None and hasattr(alphabet_model, "classes_"):
        alphabet_classes = list(alphabet_model.classes_)

    if gesture_model is not None and hasattr(gesture_model, "classes_"):
        gesture_classes = list(gesture_model.classes_)

    return jsonify({
        "status": "ASL API is running",
        "alphabet_model_loaded": alphabet_model is not None,
        "gesture_model_loaded": gesture_model is not None,
        "alphabet_model_file": ALPHABET_MODEL_FILE,
        "gesture_model_file": GESTURE_MODEL_FILE,
        "alphabet_classes": alphabet_classes,
        "gesture_classes": gesture_classes,
        "endpoints": {
            "alphabet": "/predict",
            "gesture": "/predict-gesture"
        }
    })


# =========================
# Health check
# =========================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "alphabet_model_loaded": alphabet_model is not None,
        "gesture_model_loaded": gesture_model is not None
    })


# =========================
# Predict alphabet: A-Z / del / space
# Input:
# {
#   "landmarks": [63 values]
# }
# =========================
@app.route("/predict", methods=["POST"])
def predict_alphabet():
    try:
        if alphabet_model is None:
            return jsonify({
                "error": "Alphabet model not loaded",
                "model_file": ALPHABET_MODEL_FILE
            }), 500

        data = request.get_json()

        if data is None:
            return jsonify({"error": "No JSON body received"}), 400

        if "landmarks" not in data:
            return jsonify({"error": "Missing 'landmarks' field"}), 400

        landmarks = data["landmarks"]

        if not isinstance(landmarks, list):
            return jsonify({"error": "'landmarks' must be a list"}), 400

        if len(landmarks) != 63:
            return jsonify({
                "error": f"Expected 63 landmark values, got {len(landmarks)}"
            }), 400

        X = np.array(landmarks, dtype=float).reshape(1, -1)

        prediction = alphabet_model.predict(X)[0]

        confidence = 0.0
        if hasattr(alphabet_model, "predict_proba"):
            probabilities = alphabet_model.predict_proba(X)[0]
            confidence = float(np.max(probabilities))

        return jsonify({
            "type": "alphabet",
            "prediction": str(prediction),
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# Predict gesture / phrase
# Input:
# {
#   "sequence": [
#     [63 values],
#     [63 values],
#     ...
#     [63 values]
#   ]
# }
# Total = 30 frames
# =========================
@app.route("/predict-gesture", methods=["POST"])
def predict_gesture():
    try:
        if gesture_model is None:
            return jsonify({
                "error": "Gesture model not loaded",
                "model_file": GESTURE_MODEL_FILE
            }), 500

        data = request.get_json()

        if data is None:
            return jsonify({"error": "No JSON body received"}), 400

        if "sequence" not in data:
            return jsonify({"error": "Missing 'sequence' field"}), 400

        sequence = data["sequence"]

        if not isinstance(sequence, list):
            return jsonify({"error": "'sequence' must be a list"}), 400

        if len(sequence) != 30:
            return jsonify({
                "error": f"Expected 30 frames, got {len(sequence)}"
            }), 400

        flat_sequence = []

        for frame_index, frame_landmarks in enumerate(sequence):
            if not isinstance(frame_landmarks, list):
                return jsonify({
                    "error": f"Frame {frame_index} must be a list"
                }), 400

            if len(frame_landmarks) != 63:
                return jsonify({
                    "error": (
                        f"Frame {frame_index} expected 63 values, "
                        f"got {len(frame_landmarks)}"
                    )
                }), 400

            flat_sequence.extend(frame_landmarks)

        if len(flat_sequence) != 1890:
            return jsonify({
                "error": f"Expected 1890 total values, got {len(flat_sequence)}"
            }), 400

        X = np.array(flat_sequence, dtype=float).reshape(1, -1)

        prediction = gesture_model.predict(X)[0]

        confidence = 0.0
        top3 = []

        if hasattr(gesture_model, "predict_proba"):
            probabilities = gesture_model.predict_proba(X)[0]
            classes = gesture_model.classes_

            confidence = float(np.max(probabilities))

            top_indices = np.argsort(probabilities)[::-1][:3]
            for idx in top_indices:
                label = str(classes[idx])
                top3.append({
                    "label": label,
                    "text_th": GESTURE_TEXT_TH.get(label, label),
                    "confidence": float(probabilities[idx])
                })

        prediction = str(prediction)

        return jsonify({
            "type": "gesture",
            "prediction": prediction,
            "text_th": GESTURE_TEXT_TH.get(prediction, prediction),
            "confidence": confidence,
            "top3": top3
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# Run local / Render
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)