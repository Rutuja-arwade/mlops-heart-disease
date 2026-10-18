import os
import socket
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src import __version__, config
from src.api import monitoring as mon
from src.api.schemas import BatchRequest, BatchResponse, HealthResponse, PatientFeatures, PredictionResponse
from src.predict import load_metadata, predict_records

logger = mon.configure_logging()


class ModelState:
    def __init__(self):
        self.model = None
        self.metadata = {}
        self.started = time.time()

    @property
    def version(self):
        return self.metadata.get("model_version", "unknown")

    @property
    def threshold(self):
        return float(self.metadata.get("decision_threshold", 0.5))

    def load(self):
        model_path = Path(os.getenv("MODEL_PATH", config.MODEL_FILE))
        meta_path = Path(os.getenv("MODEL_METADATA_PATH", config.MODEL_METADATA_FILE))
        try:
            self.model = joblib.load(model_path)
            self.metadata = load_metadata(meta_path)
            mon.MODEL_LOADED.set(1)
            mon.MODEL_INFO.info({"version": self.version, "family": str(self.metadata.get("model_family", "unknown")),
                                 "mlflow_run_id": str(self.metadata.get("mlflow_run_id", ""))})
            logger.info("model loaded", extra={"fields": {"path": str(model_path), "model_version": self.version}})
        except Exception:
            self.model = None
            mon.MODEL_LOADED.set(0)
            logger.exception("model failed to load", extra={"fields": {"path": str(model_path)}})


state = ModelState()


@asynccontextmanager
async def lifespan(_):
    state.load()
    yield


app = FastAPI(title="Heart Disease Risk API", version=__version__, lifespan=lifespan,
              description="Predicts heart disease from 13 clinical attributes (UCI Cleveland). Prometheus metrics at /metrics.")


def endpoint_label(request):
    return getattr(request.scope.get("route"), "path", "unmatched")


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    request.state.request_id = rid
    start = time.perf_counter()
    mon.IN_PROGRESS.inc()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
    finally:
        elapsed = time.perf_counter() - start
        mon.IN_PROGRESS.dec()
        ep = endpoint_label(request)
        if ep != "/metrics":
            mon.REQUESTS.labels(request.method, ep, str(status)).inc()
            mon.LATENCY.labels(request.method, ep).observe(elapsed)
            logger.info("request", extra={"fields": {"request_id": rid, "method": request.method, "path": request.url.path, "status": status,
                                                     "duration_ms": round(elapsed * 1000, 2), "client": request.client.host if request.client else None,
                                                     "user_agent": request.headers.get("user-agent")}})
    response.headers["X-Request-ID"] = rid
    response.headers["X-Model-Version"] = state.version
    return response


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    mon.VALIDATION_ERRORS.labels(endpoint_label(request)).inc()
    logger.warning("validation error", extra={"fields": {"request_id": getattr(request.state, "request_id", None), "errors": exc.errors()}})
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})


def require_model():
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return state.model


@app.get("/", tags=["service"])
def root():
    return {"service": "heart-disease-risk-api", "version": __version__, "model_version": state.version, "docs": "/docs",
            "endpoints": ["/health", "/ready", "/model/info", "/predict", "/predict/batch", "/metrics"], "hostname": socket.gethostname()}


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health():
    return HealthResponse(status="ok", model_loaded=state.model is not None, model_version=state.version, uptime_seconds=round(time.time() - state.started, 1))


@app.get("/ready", tags=["service"])
def ready():
    require_model()
    return {"status": "ready", "model_version": state.version}


@app.get("/model/info", tags=["model"])
def model_info():
    require_model()
    keys = ["model_name", "model_family", "model_version", "mlflow_run_id", "trained_at", "decision_threshold", "features", "best_params", "cv_metrics", "test_metrics"]
    return {k: state.metadata.get(k) for k in keys}


def score(request, records):
    model = require_model()
    t0 = time.perf_counter()
    results = predict_records(model, records, state.threshold)
    mon.INFERENCE_LATENCY.observe(time.perf_counter() - t0)
    for rec, r in zip(records, results):
        mon.PREDICTIONS.labels(r["label"], r["risk_level"]).inc()
        mon.PROBABILITY.observe(r["probability_disease"])
        logger.info("prediction", extra={"fields": {"request_id": request.state.request_id, "model_version": state.version, "prediction": r["prediction"],
                                                    "probability_disease": r["probability_disease"], "risk_level": r["risk_level"], "features": rec}})
    return results


@app.post("/predict", response_model=PredictionResponse, tags=["model"])
def predict(patient: PatientFeatures, request: Request):
    r = score(request, [patient.model_dump()])[0]
    return PredictionResponse(request_id=request.state.request_id, threshold=state.threshold, model_version=state.version, **r)


@app.post("/predict/batch", response_model=BatchResponse, tags=["model"])
def predict_batch(batch: BatchRequest, request: Request):
    results = score(request, [r.model_dump() for r in batch.records])
    return BatchResponse(request_id=request.state.request_id, model_version=state.version, count=len(results), predictions=results)


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
