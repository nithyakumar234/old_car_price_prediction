from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)
DATA_PATH = Path(__file__).resolve().parent.parent / "car_price_prediction.csv"
NUMERIC_FEATURES = [
    "Levy",
    "Prod. year",
    "Engine volume",
    "Mileage",
    "Cylinders",
    "Airbags",
    "Car_age",
]
CATEGORICAL_FEATURES = [
    "Manufacturer",
    "Category",
    "Leather interior",
    "Fuel type",
    "Gear box type",
    "Drive wheels",
    "Doors",
    "Wheel",
    "Color",
]
MODEL = None


def get_model():
    global MODEL
    if MODEL is None:
        data = pd.read_csv(DATA_PATH)
        data = data.drop(columns=["ID", "Model"], errors="ignore").drop_duplicates()
        data["Levy"] = pd.to_numeric(data["Levy"].replace("-", np.nan), errors="coerce")
        data["Mileage"] = pd.to_numeric(
            data["Mileage"].astype(str).str.replace("km", "", regex=False),
            errors="coerce",
        )
        data["Engine volume"] = pd.to_numeric(data["Engine volume"], errors="coerce")
        data = data[(data["Price"] > 100) & (data["Price"] != 26307500)].copy()
        data["Car_age"] = 2026 - data["Prod. year"]

        preprocess = ColumnTransformer(
            transformers=[
                ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
                (
                    "categorical",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    CATEGORICAL_FEATURES,
                ),
            ]
        )
        MODEL = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
            ]
        )
        MODEL.fit(data[NUMERIC_FEATURES + CATEGORICAL_FEATURES], data["Price"])
    return MODEL


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send car details as JSON."}), 400

    try:
        car = {
            "Levy": float(payload["Levy"]),
            "Manufacturer": str(payload["Manufacturer"]),
            "Prod. year": int(payload["Production year"]),
            "Category": str(payload["Category"]),
            "Leather interior": str(payload["Leather interior"]),
            "Fuel type": str(payload["Fuel type"]),
            "Engine volume": float(payload["Engine volume"]),
            "Mileage": float(payload["Mileage"]),
            "Cylinders": float(payload["Cylinders"]),
            "Gear box type": str(payload["Gear box type"]),
            "Drive wheels": str(payload["Drive wheels"]),
            "Doors": str(payload["Doors"]),
            "Wheel": str(payload["Wheel"]),
            "Color": str(payload["Color"]),
            "Airbags": float(payload["Airbags"]),
        }
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Please provide valid values for all car details."}), 400

    if not all(np.isfinite(value) for value in car.values() if isinstance(value, (int, float))):
        return jsonify({"error": "Numeric car details must be finite numbers."}), 400
    if any(value < 0 for key, value in car.items() if key != "Prod. year" and isinstance(value, (int, float))):
        return jsonify({"error": "Numeric car details cannot be negative."}), 400

    car["Car_age"] = 2026 - car["Prod. year"]
    features = pd.DataFrame([car], columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    predicted_price = float(get_model().predict(features)[0])
    return jsonify({"predicted_price": predicted_price})


if __name__ == "__main__":
    app.run(debug=True)
