from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# -----------------------------
# PATH SETUP
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

model_dir = os.path.join(BASE_DIR, "models")

# Load models
crop_model = joblib.load(os.path.join(model_dir, "crop_model.pkl"))
soil_model = joblib.load(os.path.join(model_dir, "soil_model.pkl"))
fert_model = joblib.load(os.path.join(model_dir, "fertilizer_model.pkl"))

# -----------------------------
# INIT APP
# -----------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "message": "🌱 AI Soil Analysis API is Running"
    })

# -----------------------------
# PREDICT ROUTE
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # -------- Input --------
        N = data["N"]
        P = data["P"]
        K = data["K"]
        temp = data["temperature"]
        humidity = data["humidity"]
        ph = data["ph"]
        rainfall = data.get("rainfall", 100)
        moisture = data.get("moisture", 50)
        organic_carbon = data.get("Organic_Carbon", 0.5)
        ec = data.get("Electrical_Conductivity", 1.0)

        # -------- Crop --------
        crop_input = np.array([[N, P, K, temp, humidity, ph, rainfall]])
        crop_pred = crop_model.predict(crop_input)[0]

        # -------- Soil --------
        soil_input = np.array([[temp, humidity, moisture, ph, organic_carbon, ec, N, P, K]])
        soil_pred = soil_model.predict(soil_input)[0]

        # -------- Fertilizer --------
        fert_input = np.array([[temp, humidity, moisture, N, P, K]])
        fert_pred = fert_model.predict(fert_input)[0]

        return jsonify({
            "success": True,
            "prediction": {
                "recommended_crop": crop_pred,
                "soil_type": soil_pred,
                "recommended_fertilizer": fert_pred
            }
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)