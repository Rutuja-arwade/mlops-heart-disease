import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src import config
from src.data import split_features_target
from src.evaluate import compute_metrics
from src.features import ClinicalFeatureBuilder, build_pipeline
from src.predict import predict_records, risk_level

X, y = split_features_target(pd.read_csv(config.CLEAN_DATA_FILE))
sick = {"age": 63, "sex": 1, "cp": 4, "trestbps": 150, "chol": 286, "fbs": 0, "restecg": 2, "thalach": 108, "exang": 1, "oldpeak": 1.5, "slope": 2, "ca": 3, "thal": 7}
healthy = {"age": 37, "sex": 0, "cp": 3, "trestbps": 120, "chol": 200, "fbs": 0, "restecg": 0, "thalach": 185, "exang": 0, "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3}


def test_feature_builder():
    out = ClinicalFeatureBuilder().fit_transform(X.head(5))
    assert out["hr_achieved_ratio"].iloc[0] == X["thalach"].iloc[0] / (220 - X["age"].iloc[0])
    assert "rate_pressure_index" in out.columns


def test_pipeline_cv_auc():
    pipe = build_pipeline(LogisticRegression(max_iter=2000, solver="liblinear"))
    auc = cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc").mean()
    assert auc > 0.85
    proba = pipe.fit(X, y).predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1)


def test_predict_records():
    pipe = build_pipeline(LogisticRegression(max_iter=2000, solver="liblinear")).fit(X, y)
    a, b = predict_records(pipe, [sick, healthy])
    assert a["prediction"] == 1 and b["prediction"] == 0
    assert 0.5 <= a["confidence"] <= 1
    assert risk_level(0.1) == "low" and risk_level(0.9) == "high"


def test_metrics():
    m = compute_metrics(np.array([0, 1, 1, 0]), np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.8, 0.2]))
    assert m["accuracy"] == 1.0 and m["roc_auc"] == 1.0
