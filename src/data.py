import argparse
import hashlib
import io
import json
import logging
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src import config

logger = logging.getLogger(__name__)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url, timeout=30):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    content = r.content
    if url.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            content = z.read("processed.cleveland.data")
    return content


def download_raw_data(destination=config.RAW_DATA_FILE, urls=None, force=False, retries=3):
    destination = Path(destination)
    if destination.exists() and not force:
        if sha256_of(destination) == config.RAW_DATA_SHA256:
            logger.info("Raw data already present and verified: %s", config.display_path(destination))
            return destination
        logger.warning("Checksum mismatch for %s, downloading again", config.display_path(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in urls or config.DATA_URLS:
        for attempt in range(1, retries + 1):
            try:
                logger.info("Downloading %s (attempt %d/%d)", url, attempt, retries)
                payload = _fetch(url)
                destination.write_bytes(payload)
                checksum = sha256_of(destination)
                if checksum != config.RAW_DATA_SHA256:
                    logger.warning("Downloaded file checksum %s differs from expected", checksum)
                logger.info("Saved %d bytes to %s", len(payload), config.display_path(destination))
                return destination
            except (requests.RequestException, KeyError, zipfile.BadZipFile) as e:
                errors.append(f"{url}: {e}")
                time.sleep(2 * attempt)
    raise RuntimeError("Could not download the dataset from any mirror:\n" + "\n".join(errors))


def load_raw_data(path=config.RAW_DATA_FILE):
    return pd.read_csv(path, header=None, names=config.RAW_COLUMNS, na_values="?")


def clean_data(raw):
    df = raw.copy()
    report = {"rows_in": int(len(df))}
    for c in config.RAW_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    report["missing_in_raw"] = {c: int(n) for c, n in df.isna().sum().items() if n}

    oor = {}
    for c, (lo, hi) in config.PLAUSIBLE_RANGES.items():
        m = df[c].notna() & ~df[c].between(lo, hi)
        if m.any():
            oor[c] = int(m.sum())
            df.loc[m, c] = np.nan
    report["out_of_range_set_to_nan"] = oor

    bad = {}
    for c, levels in config.CATEGORY_LEVELS.items():
        m = df[c].notna() & ~df[c].isin(levels)
        if m.any():
            bad[c] = int(m.sum())
            df.loc[m, c] = np.nan
    report["invalid_codes_set_to_nan"] = bad

    df = df.dropna(subset=["num"])
    df[config.TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])
    n = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = int(n - len(df))
    for c in config.BINARY_FEATURES + ["cp", "restecg", "slope", "ca", "thal", "age"]:
        df[c] = df[c].astype("Int64")

    report["rows_out"] = int(len(df))
    report["missing_after_cleaning"] = {c: int(n) for c, n in df.isna().sum().items() if n}
    report["class_balance"] = {str(k): int(v) for k, v in df[config.TARGET].value_counts().sort_index().items()}
    return df, report


def load_clean_data(path=config.CLEAN_DATA_FILE):
    return pd.read_csv(path)


def split_features_target(df):
    return df[config.FEATURE_ORDER].astype(float), df[config.TARGET].astype(int)


def run(force=False):
    raw_path = download_raw_data(force=force)
    clean, report = clean_data(load_raw_data(raw_path))
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(config.CLEAN_DATA_FILE, index=False)
    (config.PROCESSED_DATA_DIR / "cleaning_report.json").write_text(json.dumps(report, indent=2))
    logger.info("Clean dataset written to %s (%d rows)", config.display_path(config.CLEAN_DATA_FILE), len(clean))
    logger.info("Cleaning report: %s", json.dumps(report))
    return clean


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    run(force=args.force)
