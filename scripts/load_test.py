import argparse
import random
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config


def make_payloads(n, invalid_share, seed):
    rng = random.Random(seed)
    df = pd.read_csv(config.CLEAN_DATA_FILE).drop(columns=[config.TARGET])
    out = []
    for _ in range(n):
        row = df.sample(1, random_state=rng.randint(0, 10**6)).iloc[0].to_dict()
        rec = {k: (int(v) if k in config.CATEGORY_LEVELS or k == "age" else float(v)) for k, v in row.items() if not pd.isna(v)}
        rec["chol"] = round(min(600.0, max(110.0, rec["chol"] * rng.uniform(0.95, 1.05))), 1)
        if rng.random() < invalid_share:
            rec["age"] = rng.choice([150, -1, 5])
        out.append(rec)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--invalid-share", type=float, default=0.05)
    ap.add_argument("--host-header", default=None)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    headers = {"Host": args.host_header} if args.host_header else {}
    s = requests.Session()

    def call(payload):
        t = time.perf_counter()
        r = s.post(f"{args.url}/predict", json=payload, headers=headers, timeout=10)
        return r.status_code, time.perf_counter() - t, r.json().get("prediction") if r.status_code == 200 else None

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(call, make_payloads(args.requests, args.invalid_share, args.seed)))
    wall = time.perf_counter() - start
    lat = sorted(r[1] * 1000 for r in results)
    codes = Counter(r[0] for r in results)
    preds = Counter(r[2] for r in results if r[2] is not None)
    print(f"requests={len(results)} wall={wall:.1f}s throughput={len(results) / wall:.1f} req/s")
    print(f"status codes={dict(codes)} predictions={dict(preds)}")
    print(f"latency ms: p50={statistics.median(lat):.1f} p95={lat[int(0.95 * (len(lat) - 1))]:.1f} max={lat[-1]:.1f}")
    if codes.get(500) or codes.get(503):
        sys.exit(1)
