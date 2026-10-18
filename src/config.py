import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_FILE = RAW_DATA_DIR / "processed.cleveland.data"
CLEAN_DATA_FILE = PROCESSED_DATA_DIR / "heart_disease_clean.csv"
MODELS_DIR = ROOT_DIR / "models"
MODEL_FILE = MODELS_DIR / "model.joblib"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"
MLFLOW_MODEL_DIR = MODELS_DIR / "mlflow_model"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

DATA_URLS = ["https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
             "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"]
RAW_DATA_SHA256 = "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8"

def display_path(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT_DIR))
    except ValueError:
        return str(path)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT_DIR / 'mlflow.db'}")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "heart-disease-risk")
REGISTERED_MODEL_NAME = "heart-disease-classifier"

RAW_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
TARGET = "target"

CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]
ENGINEERED_FEATURES = ["hr_achieved_ratio", "rate_pressure_index"]
FEATURE_ORDER = [c for c in RAW_COLUMNS if c != "num"]

CATEGORY_LEVELS = {"sex": [0, 1], "cp": [1, 2, 3, 4], "fbs": [0, 1], "restecg": [0, 1, 2], "exang": [0, 1],
                   "slope": [1, 2, 3], "ca": [0, 1, 2, 3], "thal": [3, 6, 7]}
PLAUSIBLE_RANGES = {"age": (18, 100), "trestbps": (70, 250), "chol": (100, 650), "thalach": (60, 230), "oldpeak": (0.0, 7.0)}
