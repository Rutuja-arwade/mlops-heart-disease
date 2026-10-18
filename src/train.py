import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV, StratifiedKFold, train_test_split

from src import __version__, config
from src.data import load_clean_data, sha256_of, split_features_target
from src.evaluate import (CV_SCORING, compute_metrics, feature_importance, plot_calibration, plot_confusion, plot_importance,
                          plot_pr, plot_roc)
from src.features import build_pipeline

logger = logging.getLogger(__name__)


def model_specs(quick=False):
    rs = config.RANDOM_STATE
    specs = [
        {"name": "baseline_majority", "estimator": DummyClassifier(strategy="most_frequent"), "params": {"features__enabled": [False]},
         "search": "grid", "n_iter": 1, "desc": "Predicts the majority class; the floor every real model must beat."},
        {"name": "logistic_regression", "estimator": LogisticRegression(solver="liblinear", max_iter=5000, random_state=rs),
         "params": {"features__enabled": [True, False], "model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                    "model__penalty": ["l1", "l2"], "model__class_weight": [None, "balanced"]},
         "search": "grid", "n_iter": 1, "desc": "Interpretable linear baseline; L1/L2 regularisation strength tuned by grid search."},
        {"name": "random_forest", "estimator": RandomForestClassifier(random_state=rs, n_jobs=1),
         "params": {"features__enabled": [True, False], "model__n_estimators": [200, 400], "model__max_depth": [None, 4, 6, 8],
                    "model__min_samples_leaf": [1, 3, 5], "model__max_features": ["sqrt", 0.5], "model__class_weight": [None, "balanced"]},
         "search": "random", "n_iter": 30, "desc": "Bagged trees capture non-linear interactions; randomised search over depth/leaf size."},
        {"name": "gradient_boosting", "estimator": GradientBoostingClassifier(random_state=rs),
         "params": {"features__enabled": [True, False], "model__n_estimators": [100, 200, 300], "model__learning_rate": [0.01, 0.05, 0.1],
                    "model__max_depth": [2, 3], "model__subsample": [0.7, 1.0], "model__min_samples_leaf": [1, 5, 10]},
         "search": "random", "n_iter": 25, "desc": "Boosted shallow trees; randomised search over learning rate and tree size."},
    ]
    if quick:
        for s in specs:
            s["params"] = {k: v[:1] for k, v in s["params"].items()}
            s["search"], s["n_iter"] = "grid", 1
            if s["name"] == "random_forest":
                s["params"]["model__n_estimators"] = [50]
    return specs


def clean_name(n):
    return n.replace("model__", "").replace("features__enabled", "engineered_features")


def git_commit():
    if os.getenv("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config.ROOT_DIR, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def log_fig(fig, filename, save_to=None):
    mlflow.log_figure(fig, f"plots/{filename}")
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight", dpi=150)
    plt.close(fig)


def train_one(spec, X_train, y_train, X_test, y_test, cv, tags, persist=True):
    name = spec["name"]
    pipe = build_pipeline(spec["estimator"])
    kw = dict(scoring=CV_SCORING, refit="roc_auc", cv=cv, n_jobs=-1, return_train_score=True, error_score="raise")
    if spec["search"] == "random":
        search = RandomizedSearchCV(pipe, spec["params"], n_iter=spec["n_iter"], random_state=config.RANDOM_STATE, **kw)
    else:
        search = GridSearchCV(pipe, spec["params"], **kw)

    with mlflow.start_run(run_name=name, tags={**tags, "model_family": name, "stage": "tuning"}) as run:
        mlflow.set_tag("mlflow.note.content", spec["desc"])
        n_cand = spec["n_iter"] if spec["search"] == "random" else len(ParameterGrid(spec["params"]))
        mlflow.log_params({"model_family": name, "search_strategy": spec["search"], "cv_folds": config.CV_FOLDS, "refit_metric": "roc_auc", "n_candidates": n_cand})
        mlflow.log_dict({k: [str(x) for x in v] for k, v in spec["params"].items()}, "search_space.json")

        t0 = time.perf_counter()
        search.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0

        res = pd.DataFrame(search.cv_results_)
        for i, row in res.iterrows():
            with mlflow.start_run(run_name=f"{name}-trial-{i:02d}", nested=True, tags={**tags, "stage": "trial"}):
                mlflow.log_params({clean_name(k): str(v) for k, v in row["params"].items()})
                m = {"rank_roc_auc": float(row["rank_test_roc_auc"])}
                for metric in CV_SCORING:
                    m[f"cv_{metric}_mean"] = float(row[f"mean_test_{metric}"])
                    m[f"cv_{metric}_std"] = float(row[f"std_test_{metric}"])
                m["train_roc_auc_mean"] = float(row["mean_train_roc_auc"])
                mlflow.log_metrics(m)

        bi = search.best_index_
        cv_metrics = {}
        for metric in CV_SCORING:
            cv_metrics[f"cv_{metric}_mean"] = float(res.loc[bi, f"mean_test_{metric}"])
            cv_metrics[f"cv_{metric}_std"] = float(res.loc[bi, f"std_test_{metric}"])

        best = search.best_estimator_
        y_proba = best.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        test_metrics = {f"test_{k}": v for k, v in compute_metrics(y_test, y_pred, y_proba).items()}
        best_params = {clean_name(k): v for k, v in search.best_params_.items()}
        mlflow.log_params({f"best_{k}": str(v) for k, v in best_params.items()})
        mlflow.log_metrics({**cv_metrics, **test_metrics, "search_fit_seconds": fit_seconds})

        title = name.replace("_", " ").title()
        fdir = config.FIGURES_DIR / "models" / name
        out = lambda f: fdir / f if persist else None
        log_fig(plot_roc(y_test, y_proba, f"ROC - {title}"), "roc_curve.png", out("roc_curve.png"))
        log_fig(plot_pr(y_test, y_proba, f"Precision-Recall - {title}"), "pr_curve.png", out("pr_curve.png"))
        log_fig(plot_confusion(y_test, y_pred, f"Confusion matrix - {title}"), "confusion_matrix.png", out("confusion_matrix.png"))
        log_fig(plot_calibration(y_test, y_proba, f"Calibration - {title}"), "calibration.png", out("calibration.png"))
        imp = feature_importance(best)
        if imp is not None:
            log_fig(plot_importance(imp, f"Top features - {title}"), "feature_importance.png", out("feature_importance.png"))
            mlflow.log_dict(imp.round(5).to_dict(), "feature_importance.json")

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "cv_results.csv"
            res.drop(columns=["params"]).to_csv(p, index=False)
            mlflow.log_artifact(str(p), artifact_path="tuning")
        mlflow.log_text(classification_report(y_test, y_pred, target_names=["no_disease", "disease"], zero_division=0), "classification_report.txt")
        mlflow.sklearn.log_model(best, artifact_path="model", signature=infer_signature(X_train, best.predict_proba(X_train)), input_example=X_train.head(3))

        logger.info("%-20s cv_roc_auc=%.3f±%.3f test_roc_auc=%.3f test_acc=%.3f best=%s", name, cv_metrics["cv_roc_auc_mean"],
                    cv_metrics["cv_roc_auc_std"], test_metrics["test_roc_auc"], test_metrics["test_accuracy"], best_params)
        return {"name": name, "pipeline": best, "best_params": best_params, "cv_metrics": cv_metrics, "test_metrics": test_metrics,
                "run_id": run.info.run_id, "y_proba": y_proba}


def comparison_figures(results, y_test):
    colors = ["#9e9e9e", "#264653", "#2a9d8f", "#e76f51", "#e9c46a"]
    roc_fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for c, r in zip(colors, results):
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        ax.plot(fpr, tpr, lw=2, color=c, label=f"{r['name']} (AUC {r['test_metrics']['test_roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Hold-out ROC curves - all models")
    ax.legend(loc="lower right", fontsize=9)

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    table = pd.DataFrame({r["name"]: [r["cv_metrics"][f"cv_{m}_mean"] for m in metrics] for r in results}, index=metrics)
    bar_fig, ax2 = plt.subplots(figsize=(9, 4.8))
    table.plot(kind="bar", ax=ax2, color=colors[:len(results)], width=0.8)
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("Mean 5-fold CV score")
    ax2.set_title("Cross-validated metrics by model")
    ax2.set_xticklabels(metrics, rotation=0)
    ax2.legend(fontsize=9, loc="lower right")
    return roc_fig, bar_fig


def export_champion(champ, X_train, results, tags, register):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(champ["pipeline"], config.MODEL_FILE, compress=3)
    version = None
    if register:
        v = mlflow.register_model(f"runs:/{champ['run_id']}/model", config.REGISTERED_MODEL_NAME)
        client = MlflowClient()
        client.set_registered_model_alias(config.REGISTERED_MODEL_NAME, "champion", v.version)
        client.update_model_version(config.REGISTERED_MODEL_NAME, v.version, description=f"{champ['name']}; cv_roc_auc={champ['cv_metrics']['cv_roc_auc_mean']:.3f}")
        version = v.version
    if config.MLFLOW_MODEL_DIR.exists():
        shutil.rmtree(config.MLFLOW_MODEL_DIR)
    mlflow.sklearn.save_model(champ["pipeline"], str(config.MLFLOW_MODEL_DIR), signature=infer_signature(X_train, champ["pipeline"].predict_proba(X_train)),
                              input_example=X_train.head(3))
    meta = {
        "model_name": config.REGISTERED_MODEL_NAME,
        "model_family": champ["name"],
        "model_version": f"{__version__}+{champ['name']}",
        "mlflow_run_id": champ["run_id"],
        "mlflow_registry_version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": tags.get("git_commit"),
        "dataset_sha256": tags.get("dataset_sha256"),
        "decision_threshold": 0.5,
        "features": config.FEATURE_ORDER,
        "best_params": {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)) for k, v in champ["best_params"].items()},
        "cv_metrics": champ["cv_metrics"],
        "test_metrics": champ["test_metrics"],
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "all_models": {r["name"]: {**r["cv_metrics"], **r["test_metrics"]} for r in results},
    }
    config.MODEL_METADATA_FILE.write_text(json.dumps(meta, indent=2))
    return meta


def run(quick=False, register=True):
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    X, y = split_features_target(load_clean_data())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    logger.info("train=%d test=%d positive_rate=%.3f", len(X_train), len(X_test), y.mean())
    tags = {"git_commit": git_commit(), "dataset_sha256": sha256_of(config.CLEAN_DATA_FILE), "quick_mode": str(quick),
            "triggered_by": "github-actions" if os.getenv("GITHUB_ACTIONS") else "local"}

    results = [train_one(s, X_train, y_train, X_test, y_test, cv, tags, persist=not quick) for s in model_specs(quick)]
    champ = max([r for r in results if not r["name"].startswith("baseline")], key=lambda r: (r["cv_metrics"]["cv_roc_auc_mean"], r["cv_metrics"]["cv_accuracy_mean"]))
    logger.info("Champion model: %s (cv_roc_auc=%.3f)", champ["name"], champ["cv_metrics"]["cv_roc_auc_mean"])

    rows = [{"model": r["name"], **r["cv_metrics"], **r["test_metrics"], "best_params": json.dumps(r["best_params"], default=str)} for r in results]
    comparison = pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False)

    with mlflow.start_run(run_name="model-selection", tags={**tags, "stage": "selection"}):
        mlflow.log_params({"champion": champ["name"], "selection_metric": "cv_roc_auc_mean"})
        mlflow.log_metrics({f"champion_{k}": v for k, v in {**champ["cv_metrics"], **champ["test_metrics"]}.items()})
        roc_fig, bar_fig = comparison_figures(results, y_test)
        log_fig(roc_fig, "roc_comparison.png", None if quick else config.FIGURES_DIR / "09_roc_comparison.png")
        log_fig(bar_fig, "cv_metric_comparison.png", None if quick else config.FIGURES_DIR / "10_cv_metric_comparison.png")
        mlflow.log_table(comparison, "model_comparison.json")
        if quick:
            return {"model_family": champ["name"], "test_metrics": champ["test_metrics"]}
        meta = export_champion(champ, X_train, results, tags, register)
        mlflow.log_dict(meta, "champion_metadata.json")

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(config.METRICS_DIR / "model_comparison.csv", index=False)
    (config.METRICS_DIR / "champion_metrics.json").write_text(json.dumps({"champion": champ["name"], **champ["cv_metrics"], **champ["test_metrics"]}, indent=2))
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-register", action="store_true")
    args = p.parse_args()
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    logger.disabled = False
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    meta = run(quick=args.quick, register=not args.no_register)
    print(json.dumps({"champion": meta["model_family"], **meta["test_metrics"]}, indent=2))
