# Car Price Predictor

A web application that predicts car prices using machine learning (XGBoost) based on user input features. The app is built with Flask and provides an easy-to-use interface for users to estimate the price of a car.

## Features

- Predicts car prices based on various input features (e.g., year, mileage, fuel type, etc.)
- User-friendly web interface
- Model trained on Cardekho dataset
- Uses XGBoost for accurate predictions

## Project Structure

```
Car Price Predictor/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── data/
│   └── cardekho_imputated.csv   # Cleaned dataset
├── artifacts/
│   └── xgb_car_price.joblib     # Trained XGBoost model
├── templates/
│   ├── home.html           # Home page template
│   └── predict.html        # Prediction result template
├── NoteBook/
│   └── XGBoost_CarPrice_Notebook.ipynb  # Model training notebook
└── README.md
```

## Getting Started

1. **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd "Car Price Predictor"
    ```

2. **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the application:**
    ```bash
    python app.py
    ```

5. **Open your browser and go to** `http://127.0.0.1:5000/`

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

- [XGBoost](https://xgboost.ai/)
- [Flask](https://flask.palletsprojects.com/)
- [Cardekho Dataset](https://www.cardekho.com/)