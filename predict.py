
from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
with open("model/heart_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "Heart Disease Prediction API. POST to /predict"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    return jsonify({
        "prediction": int(pred),
        "probability": float(prob),
        "has_heart_disease": bool(pred)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)