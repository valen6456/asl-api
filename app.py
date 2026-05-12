from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

MODEL_FILE = "landmark_model.pkl"

# โหลดโมเดล
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ASL landmark model API is running",
        "classes": list(model.classes_)
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # เช็กว่ามี JSON ส่งมาหรือไม่
        if not data:
            return jsonify({
                "error": "No JSON data received"
            }), 400

        # เช็กว่ามี key ชื่อ landmarks หรือไม่
        if "landmarks" not in data:
            return jsonify({
                "error": "Missing 'landmarks' key"
            }), 400

        landmarks = data["landmarks"]

        # เช็กว่า landmarks เป็น list หรือไม่
        if not isinstance(landmarks, list):
            return jsonify({
                "error": "'landmarks' must be a list"
            }), 400

        # โมเดลมือเดียวต้องใช้ 21 จุด x,y,z = 63 ค่า
        if len(landmarks) != 63:
            return jsonify({
                "error": f"Expected 63 landmark values, got {len(landmarks)}"
            }), 400

        # แปลงเป็นตัวเลข
        try:
            landmarks = [float(x) for x in landmarks]
        except ValueError:
            return jsonify({
                "error": "All landmark values must be numbers"
            }), 400

        X = np.array([landmarks])

        prediction = model.predict(X)[0]

        confidence = 1.0

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[0]
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
    app.run(host="0.0.0.0", port=5000)