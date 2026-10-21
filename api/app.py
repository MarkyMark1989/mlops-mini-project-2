"""Flask REST API for the churn model. Swagger UI is served at /apidocs/."""
import math
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from flasgger import Swagger
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parent.parent
# The pickled pipeline references the `features` module in src/.
sys.path.insert(0, str(ROOT / "src"))
from features import BINARY_FEATURES, RAW_FEATURES  # noqa: E402

MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "outputs" / "model" / "model.pkl"))

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Descriptions from DataDictionary-1.xlsx; examples are the first row of the dataset.
FIELD_DOCS = {
    "AccountWeeks": ("number of weeks customer has had active account", 128),
    "ContractRenewal": ("1 if customer recently renewed contract, 0 if not", 1),
    "DataPlan": ("1 if customer has data plan, 0 if not", 1),
    "DataUsage": ("gigabytes of monthly data usage", 2.7),
    "CustServCalls": ("number of calls into customer service", 1),
    "DayMins": ("average daytime minutes per month", 265.1),
    "DayCalls": ("average number of daytime calls", 110),
    "MonthlyCharge": ("average monthly bill", 89),
    "OverageFee": ("largest overage fee in last 12 months", 9.87),
    "RoamMins": ("roaming usage mins per month", 10),
}

app = Flask(__name__)
app.json.sort_keys = False
Swagger(app, template={
    "info": {
        "title": "Telecom Churn Prediction API",
        "description": "Random forest churn classifier trained on the telecom churn dataset.",
        "version": "1.0",
    },
    "definitions": {
        "Customer": {
            "type": "object",
            "required": RAW_FEATURES,
            "properties": {
                name: {"type": "number", "description": desc, "example": example}
                for name, (desc, example) in FIELD_DOCS.items()
            },
        },
    },
})


def parse_customer(payload):
    """Validate a JSON body and return it as a one-row DataFrame of raw features."""
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object of predictor values")
    missing = [col for col in RAW_FEATURES if col not in payload]
    if missing:
        raise ValueError(f"Missing predictor(s): {', '.join(missing)}")

    row = {}
    for col in RAW_FEATURES:
        value = payload[col]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{col} must be a number")
        if value < 0:
            raise ValueError(f"{col} cannot be negative")
        if col in BINARY_FEATURES and value not in (0, 1):
            raise ValueError(f"{col} must be 0 or 1")
        row[col] = float(value)
    return pd.DataFrame([row], columns=RAW_FEATURES)


@app.get("/models/parameters/")
def model_parameters():
    """Return the hyperparameters of the trained model.
    ---
    tags: [Model]
    responses:
      200:
        description: Model type and its hyperparameters (from model.get_params()).
    """
    classifier = model.named_steps["classifier"]
    return jsonify({
        "model_type": type(classifier).__name__,
        "parameters": classifier.get_params(),
    })


@app.post("/models/predictions/")
def predict():
    """Predict whether a customer will churn.
    ---
    tags: [Model]
    parameters:
      - in: body
        name: customer
        required: true
        schema:
          $ref: '#/definitions/Customer'
    responses:
      200:
        description: Predicted class (1 = churn) and churn probability.
      400:
        description: Invalid or missing predictor values.
    """
    try:
        X = parse_customer(request.get_json(silent=True))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0, 1])
    return jsonify({
        "prediction": prediction,
        "prediction_label": "churn" if prediction == 1 else "no churn",
        "churn_probability": round(probability, 4),
    })


if __name__ == "__main__":
    # Port 5000 is taken by AirPlay Receiver on macOS.
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5001)))
