import json

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src import config
from src.data import split_features_target
from src.features import build_pipeline

patient = {"age": 63, "sex": 1, "cp": 4, "trestbps": 150, "chol": 286, "fbs": 0, "restecg": 2, "thalach": 108, "exang": 1, "oldpeak": 1.5, "slope": 2, "ca": 3, "thal": 7}


@pytest.fixture
def client(tmp_path, monkeypatch):
    X, y = split_features_target(pd.read_csv(config.CLEAN_DATA_FILE))
    joblib.dump(build_pipeline(LogisticRegression(max_iter=2000, solver="liblinear")).fit(X, y), tmp_path / "m.joblib")
    (tmp_path / "meta.json").write_text(json.dumps({"model_version": "test-1", "decision_threshold": 0.5}))
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "m.joblib"))
    monkeypatch.setenv("MODEL_METADATA_PATH", str(tmp_path / "meta.json"))
    from src.api.app import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["model_loaded"] is True
    assert client.get("/ready").status_code == 200


def test_predict(client):
    r = client.post("/predict", json=patient)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == 1 and 0.5 <= body["confidence"] <= 1
    assert r.headers["X-Model-Version"] == "test-1"


def test_bad_input(client):
    assert client.post("/predict", json={**patient, "cp": 7}).status_code == 422
    assert client.post("/predict", json={**patient, "age": 300}).status_code == 422


def test_metrics(client):
    client.post("/predict", json=patient)
    txt = client.get("/metrics").text
    assert "heart_api_requests_total" in txt and "heart_api_predictions_total" in txt
