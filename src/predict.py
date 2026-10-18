import argparse
import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src import config


def risk_level(p):
    if p < 0.3:
        return "low"
    if p < 0.6:
        return "moderate"
    return "high"


@lru_cache(maxsize=4)
def load_model(path=str(config.MODEL_FILE)):
    return joblib.load(path)


def load_metadata(path=config.MODEL_METADATA_FILE):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def to_frame(records):
    df = pd.DataFrame.from_records(records)
    for c in config.FEATURE_ORDER:
        if c not in df.columns:
            df[c] = float("nan")
    return df[config.FEATURE_ORDER].astype(float)


def predict_records(model, records, threshold=0.5):
    out = []
    for p in model.predict_proba(to_frame(records))[:, 1]:
        pred = int(p >= threshold)
        out.append({"prediction": pred, "label": "heart_disease" if pred else "no_heart_disease", "probability_disease": round(float(p), 4),
                    "confidence": round(float(p if pred else 1 - p), 4), "risk_level": risk_level(float(p))})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", default=str(config.MODEL_FILE))
    args = ap.parse_args()
    payload = json.loads(Path(args.input).read_text())
    records = payload if isinstance(payload, list) else [payload]
    print(json.dumps(predict_records(load_model(args.model), records, load_metadata().get("decision_threshold", 0.5)), indent=2))
