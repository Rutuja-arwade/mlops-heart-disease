import json
import logging
import os
import sys
from datetime import datetime, timezone

from prometheus_client import Counter, Gauge, Histogram, Info

REQUESTS = Counter("heart_api_requests_total", "HTTP requests handled", ["method", "endpoint", "status_code"])
LATENCY = Histogram("heart_api_request_duration_seconds", "End-to-end HTTP request latency", ["method", "endpoint"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5))
IN_PROGRESS = Gauge("heart_api_requests_in_progress", "Requests currently being processed")
PREDICTIONS = Counter("heart_api_predictions_total", "Predictions served by outcome", ["label", "risk_level"])
PROBABILITY = Histogram("heart_api_prediction_probability", "Distribution of predicted disease probability",
                        buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
INFERENCE_LATENCY = Histogram("heart_api_inference_duration_seconds", "Model predict_proba latency",
                              buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25))
VALIDATION_ERRORS = Counter("heart_api_validation_errors_total", "Rejected requests (HTTP 422)", ["endpoint"])
MODEL_LOADED = Gauge("heart_api_model_loaded", "1 if the model is loaded and ready")
MODEL_INFO = Info("heart_api_model", "Metadata about the served model")


class JsonFormatter(logging.Formatter):
    def format(self, record):
        d = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        if getattr(record, "fields", None):
            d.update(record.fields)
        if record.exc_info:
            d["exception"] = self.formatException(record.exc_info)
        return json.dumps(d, default=str)


def configure_logging():
    logger = logging.getLogger("heart_api")
    if logger.handlers:
        return logger
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    logger.addHandler(h)
    logger.propagate = False
    log_file = os.getenv("LOG_FILE")
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(JsonFormatter())
            logger.addHandler(fh)
        except OSError as e:
            logger.warning("file logging disabled", extra={"fields": {"log_file": log_file, "error": str(e)}})
    return logger
