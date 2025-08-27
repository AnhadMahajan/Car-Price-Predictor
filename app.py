from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os

# Load the saved model pipeline
MODEL_PATH = "artifacts/xgb_car_price.joblib"
model_artifact = joblib.load(MODEL_PATH)
pipeline = model_artifact["pipeline"]
feature_cols = model_artifact["feature_cols"]

app = Flask(__name__)

def preprocess_input(form_data):
    """Convert form inputs into dataframe row consistent with training"""
    row = pd.DataFrame([{col: np.nan for col in feature_cols}])

    # Fill values from form
    for k, v in form_data.items():
        if k in row.columns:
            row.at[0, k] = v

    # Normalize mileage, engine, max_power if needed
    for col in ["mileage", "engine", "max_power"]:
        if col in row.columns and pd.notna(row.at[0, col]):
            try:
                row.at[0, col] = float(str(row.at[0, col]).split()[0])
            except:
                pass

    # Cast numeric fields if possible
    for col in row.columns:
        try:
            row[col] = pd.to_numeric(row[col])
        except:
            pass

    return row

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        form_data = request.form.to_dict()
        input_df = preprocess_input(form_data)
        prediction = pipeline.predict(input_df)[0]
        return render_template("predict.html",
                               prediction=round(prediction, 2),
                               data=form_data)
    return None


if __name__ == "__main__":
    app.run(debug=True)
