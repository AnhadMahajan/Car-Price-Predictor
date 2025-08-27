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

from datetime import datetime

def preprocess_input(form_data):
    row = pd.DataFrame([{col: None for col in feature_cols}], dtype=object)

    for k, v in form_data.items():
        if k in row.columns:
            row.at[0, k] = v

    numeric_cols = ["year", "km_driven", "mileage", "engine", "max_power", "seats"]
    for col in numeric_cols:
        if col in row.columns and pd.notna(row.at[0, col]):
            try:
                row.at[0, col] = float(str(row.at[0, col]).split()[0])
            except:
                row.at[0, col] = np.nan

    # ✅ calculate vehicle age if required
    if "vehicle_age" in row.columns and "year" in form_data:
        current_year = datetime.now().year
        try:
            row["vehicle_age"] = current_year - int(form_data["year"])
        except:
            row["vehicle_age"] = np.nan

    # Drop unwanted columns
    drop_cols = ["car_name", "brand", "year"]  # drop 'year' if only using vehicle_age
    row = row.drop(columns=[c for c in drop_cols if c in row.columns], errors="ignore")

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
        print(f"✅ Predicted Price: {round(prediction, 2)}")

        print("Form Data:", form_data)
        print("Processed Input DF:")
        print(input_df.head())

        return render_template("predict.html",
                               prediction=round(prediction, 2),
                               data=form_data)
    return None


if __name__ == "__main__":
    app.run(debug=True)
