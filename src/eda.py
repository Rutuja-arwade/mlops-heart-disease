import json
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config
from src.data import load_clean_data

logger = logging.getLogger(__name__)

PALETTE = {0: "#2a9d8f", 1: "#e76f51"}
LABELS = {0: "No disease", 1: "Disease"}
PRETTY = {"age": "Age (years)", "trestbps": "Resting BP (mm Hg)", "chol": "Serum cholesterol (mg/dl)", "thalach": "Max heart rate (bpm)",
          "oldpeak": "ST depression (oldpeak)", "ca": "Major vessels (ca)", "cp": "Chest pain type", "restecg": "Resting ECG",
          "slope": "ST slope", "thal": "Thalassemia", "sex": "Sex (1 = male)", "fbs": "Fasting sugar >120", "exang": "Exercise angina"}
T = config.TARGET


def save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved %s", config.display_path(path))
    return config.display_path(path)


def plot_class_balance(df):
    counts = df[T].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([LABELS[i] for i in counts.index], counts.values, color=[PALETTE[i] for i in counts.index])
    for b, v in zip(bars, counts.values):
        ax.annotate(f"{v}  ({v / counts.sum():.1%})", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Patients"); ax.set_title("Class balance of the target")
    ax.set_ylim(0, counts.max() * 1.18)
    return save(fig, "01_class_balance.png")


def plot_missing(raw_missing, n_rows):
    cols = config.FEATURE_ORDER
    vals = [raw_missing.get(c, 0) for c in cols]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(cols, vals, color=["#e9c46a" if v else "#d9d9d9" for v in vals])
    for i, v in enumerate(vals):
        if v:
            ax.annotate(f"{v} ({v / n_rows:.1%})", (i, v), ha="center", va="bottom")
    ax.set_title("Missing values per feature in the raw file ('?' markers)")
    ax.set_ylabel("Missing rows")
    ax.set_ylim(0, max(vals + [1]) * 1.4)
    return save(fig, "02_missing_values.png")


def plot_histograms(df):
    cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, col in zip(axes.ravel(), cols):
        for lab in (0, 1):
            d = col == "ca"
            sns.histplot(df.loc[df[T] == lab, col].dropna(), ax=ax, color=PALETTE[lab], label=LABELS[lab], kde=not d,
                         stat="density", alpha=0.45, bins=4 if d else 18, discrete=d)
        ax.set_title(PRETTY[col]); ax.set_xlabel("")
    axes[0, 0].legend()
    fig.suptitle("Distribution of continuous features by diagnosis", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return save(fig, "03_feature_histograms.png")


def plot_correlation(df):
    corr = df.astype(float).corr(method="pearson")
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool), k=1), cmap="vlag", center=0, annot=True, fmt=".2f",
                annot_kws={"size": 8}, linewidths=0.5, cbar_kws={"shrink": 0.75, "label": "Pearson r"}, ax=ax)
    ax.set_title("Correlation heatmap (features and target)")
    return save(fig, "04_correlation_heatmap.png")


def plot_target_correlation(df):
    corr = df.astype(float).corr()[T].drop(T).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(corr.index, corr.values, color=["#2a9d8f" if v < 0 else "#e76f51" for v in corr.values])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Correlation of each feature with the target")
    ax.set_xlabel("Pearson r with disease presence")
    return save(fig, "05_target_correlation.png")


def plot_categorical_rates(df):
    cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    for ax, col in zip(axes.ravel(), cols):
        tab = pd.crosstab(df[col], df[T], normalize="index")
        counts = df[col].value_counts().sort_index()
        tab.plot(kind="bar", stacked=True, ax=ax, color=[PALETTE[0], PALETTE[1]], legend=False, width=0.75)
        ax.set_title(PRETTY[col]); ax.set_xlabel(""); ax.set_ylabel("Share of patients")
        ax.set_xticklabels([f"{int(float(i))}\n(n={counts[i]})" for i in tab.index], rotation=0)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=PALETTE[i]) for i in (0, 1)], [LABELS[0], LABELS[1]], loc="upper right")
    fig.suptitle("Disease rate within each category", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return save(fig, "06_categorical_disease_rate.png")


def plot_boxplots(df):
    cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    long = df.melt(id_vars=T, value_vars=cols)
    long["Diagnosis"] = long[T].map(LABELS)
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.2))
    for ax, col in zip(axes, cols):
        sns.boxplot(data=long[long["variable"] == col], x="Diagnosis", y="value", hue="Diagnosis",
                    palette={LABELS[0]: PALETTE[0], LABELS[1]: PALETTE[1]}, ax=ax, legend=False)
        ax.set_title(PRETTY[col]); ax.set_xlabel(""); ax.set_ylabel("")
    fig.suptitle("Outlier view: continuous features by diagnosis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return save(fig, "07_boxplots.png")


def plot_age_vs_hr(df):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for lab in (0, 1):
        s = df[df[T] == lab]
        ax.scatter(s["age"], s["thalach"], c=PALETTE[lab], label=LABELS[lab], alpha=0.7, edgecolor="white")
    ages = np.linspace(df["age"].min(), df["age"].max(), 50)
    ax.plot(ages, 220 - ages, "k--", lw=1, label="Age-predicted max HR (220 - age)")
    ax.set_xlabel("Age (years)"); ax.set_ylabel("Max heart rate achieved (bpm)")
    ax.set_title("Heart-rate response vs age (motivates hr_achieved_ratio)")
    ax.legend()
    return save(fig, "08_age_vs_max_hr.png")


def summary_tables(df):
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df.describe().T.round(2).to_csv(config.METRICS_DIR / "eda_summary_statistics.csv")
    by_class = df.groupby(T).mean(numeric_only=True).T.round(2)
    by_class.columns = [LABELS[c] for c in by_class.columns]
    by_class.to_csv(config.METRICS_DIR / "eda_means_by_class.csv")
    corr = df.astype(float).corr()[T].drop(T).sort_values(key=np.abs, ascending=False)
    return {"rows": int(len(df)), "features": len(config.FEATURE_ORDER),
            "class_counts": {LABELS[int(k)]: int(v) for k, v in df[T].value_counts().sort_index().items()},
            "top_target_correlations": corr.head(6).round(3).to_dict()}


def run():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.titleweight": "bold"})
    df = load_clean_data()
    rp = config.PROCESSED_DATA_DIR / "cleaning_report.json"
    cleaning = json.loads(rp.read_text()) if rp.exists() else {}
    figs = [plot_class_balance(df), plot_missing(cleaning.get("missing_in_raw", {}), len(df)), plot_histograms(df), plot_correlation(df),
            plot_target_correlation(df), plot_categorical_rates(df), plot_boxplots(df), plot_age_vs_hr(df)]
    summary = summary_tables(df)
    summary["figures"] = figs
    (config.METRICS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("EDA summary: %s", json.dumps({k: v for k, v in summary.items() if k != "figures"}))
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    run()
