"""O2 - Fixed-time baseline + stability check.

Runs the fixed-time control plan across three seeds and reports the coefficient
of variation (CV = sigma / mu) for each metric. Acceptance criterion (O2):
CV <= 5% for delay, queue and CO2, confirming the baseline is a stable control
condition for the O3/O4 comparison.

Outputs (in ./results):
  baseline_runs.csv  - per-seed metrics
  baseline_cv.csv    - mean, std, CV and pass/fail per metric
"""
import os

import numpy as np
import pandas as pd

from sim import run_simulation

# Fixed-time baseline: 60 s cycle (27 s green each way + 3 s yellow each way).
BASELINE_NS_GREEN = 27
BASELINE_EW_GREEN = 27
SEEDS = [42, 123, 999]
CV_THRESHOLD = 0.05  # 5%

# Metrics the O2 acceptance criterion is checked against.
CV_METRICS = ["waiting_time", "mean_queue", "co2"]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def run_baseline(seeds=SEEDS, ns_green=BASELINE_NS_GREEN, ew_green=BASELINE_EW_GREEN):
    """Run the baseline plan once per seed; return a DataFrame of per-run metrics."""
    rows = []
    for seed in seeds:
        m = run_simulation(ns_green, ew_green, seed=seed, label=f"baseline_{seed}")
        rows.append(m)
        print(f"  seed {seed:>4}: waiting={m['waiting_time']:.0f}  "
              f"co2={m['co2']:.0f}  mean_queue={m['mean_queue']:.3f}  "
              f"throughput={m['throughput']}")
    return pd.DataFrame(rows)


def compute_cv(df, metrics=CV_METRICS):
    """Return a DataFrame of mean, sample std, CV and pass/fail per metric."""
    out = []
    for metric in metrics:
        values = df[metric].to_numpy(dtype=float)
        mean = values.mean()
        std = values.std(ddof=1)  # sample std (n-1) for replications
        cv = std / mean if mean else float("nan")
        out.append({
            "metric": metric,
            "mean": mean,
            "std": std,
            "cv": cv,
            "cv_pct": cv * 100,
            "passes_5pct": bool(cv <= CV_THRESHOLD),
        })
    return pd.DataFrame(out)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"O2 baseline: plan NS={BASELINE_NS_GREEN}s EW={BASELINE_EW_GREEN}s "
          f"(cycle {BASELINE_NS_GREEN + BASELINE_EW_GREEN + 6}s), seeds {SEEDS}")

    runs = run_baseline()
    runs_path = os.path.join(RESULTS_DIR, "baseline_runs.csv")
    runs.to_csv(runs_path, index=False)

    cv = compute_cv(runs)
    cv_path = os.path.join(RESULTS_DIR, "baseline_cv.csv")
    cv.to_csv(cv_path, index=False)

    print("\nCV stability (acceptance: CV <= 5%):")
    for _, r in cv.iterrows():
        status = "PASS" if r["passes_5pct"] else "FAIL"
        print(f"  {r['metric']:<13} mean={r['mean']:.1f}  "
              f"CV={r['cv_pct']:.2f}%  [{status}]")

    overall = "PASS" if cv["passes_5pct"].all() else "FAIL"
    print(f"\nO2 overall: {overall}")
    print(f"Saved: {runs_path}\n       {cv_path}")
    return runs, cv


if __name__ == "__main__":
    main()
