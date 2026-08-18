"""O4 - Paired evaluation of the evolved plan vs the fixed-time baseline.

Runs both plans across five matched seeds, tests whether the GA plan
significantly reduces delay and CO2 (Wilcoxon signed-rank, one-tailed), reports
the percentage improvement and a rank-biserial effect size (the correct companion
to Wilcoxon - see WS4), and saves box plots.

Acceptance (O4): >=15% delay and >=10% CO2 reduction, both at p < 0.05.

Outputs (in ./results):
  eval_paired_runs.csv - per-seed metrics for both conditions
  eval_stats.csv       - per-metric improvement, Wilcoxon p, effect size, pass/re-evaluate
  eval_boxplots.png    - delay & CO2 box plots (baseline vs GA)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from sim import run_simulation
from baseline import BASELINE_NS_GREEN, BASELINE_EW_GREEN

EVAL_SEEDS = [42, 123, 999, 7, 2024]
THRESHOLDS = {"waiting_time": 0.15, "co2": 0.10}  # required fractional reduction
ALPHA = 0.05

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _load_best_plan():
    best_csv = os.path.join(RESULTS_DIR, "ga_best.csv")
    if not os.path.exists(best_csv):
        raise SystemExit("Run ga_optimise.py first (results/ga_best.csv missing).")
    row = pd.read_csv(best_csv).iloc[0]
    return int(row["ns_green"]), int(row["ew_green"])


def run_paired(ga_ns, ga_ew, seeds=EVAL_SEEDS):
    rows = []
    for seed in seeds:
        b = run_simulation(BASELINE_NS_GREEN, BASELINE_EW_GREEN, seed=seed,
                           label=f"eval_base_{seed}")
        b["condition"] = "baseline"
        g = run_simulation(ga_ns, ga_ew, seed=seed, label=f"eval_ga_{seed}")
        g["condition"] = "ga"
        rows.extend([b, g])
        print(f"  seed {seed:>4}: base_wait={b['waiting_time']:.0f} "
              f"ga_wait={g['waiting_time']:.0f} | base_co2={b['co2']:.0f} "
              f"ga_co2={g['co2']:.0f}")
    return pd.DataFrame(rows)


def rank_biserial(baseline_vals, ga_vals):
    """Matched-pairs rank-biserial correlation for the Wilcoxon signed-rank test.

    Positive = baseline larger than GA (i.e. GA improves). Range [-1, 1].
    """
    diffs = np.asarray(baseline_vals) - np.asarray(ga_vals)
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diffs))
    t_plus = ranks[diffs > 0].sum()
    t_minus = ranks[diffs < 0].sum()
    return (t_plus - t_minus) / ranks.sum()


def analyse(df):
    base = df[df["condition"] == "baseline"].sort_values("seed")
    ga = df[df["condition"] == "ga"].sort_values("seed")
    out = []
    for metric, thresh in THRESHOLDS.items():
        b = base[metric].to_numpy(float)
        g = ga[metric].to_numpy(float)
        improvement = (b.mean() - g.mean()) / b.mean()
        # one-tailed: is baseline > GA (a reduction)?
        try:
            w_stat, p = stats.wilcoxon(b, g, alternative="greater")
        except ValueError:
            w_stat, p = float("nan"), float("nan")
        out.append({
            "metric": metric,
            "baseline_mean": b.mean(),
            "ga_mean": g.mean(),
            "improvement_pct": improvement * 100,
            "threshold_pct": thresh * 100,
            "wilcoxon_stat": w_stat,
            "p_value": p,
            "rank_biserial": rank_biserial(b, g),
            "passes": bool(improvement >= thresh and p < ALPHA),
        })
    return pd.DataFrame(out)


def make_boxplots(df):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, metric, title in zip(axes, ["waiting_time", "co2"],
                                 ["Total waiting time (veh·s)", "Total CO2 (mg)"]):
        data = [df[df.condition == c][metric] for c in ("baseline", "ga")]
        ax.boxplot(data, tick_labels=["baseline", "GA"])
        ax.set_title(title)
    fig.suptitle("O4: GA vs fixed-time baseline (5 paired seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "eval_boxplots.png"), dpi=120)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ga_ns, ga_ew = _load_best_plan()
    print(f"O4 evaluation: GA plan NS={ga_ns}s EW={ga_ew}s vs "
          f"baseline NS={BASELINE_NS_GREEN}s EW={BASELINE_EW_GREEN}s, seeds {EVAL_SEEDS}")

    runs = run_paired(ga_ns, ga_ew)
    runs.to_csv(os.path.join(RESULTS_DIR, "eval_paired_runs.csv"), index=False)

    stats_df = analyse(runs)
    stats_df.to_csv(os.path.join(RESULTS_DIR, "eval_stats.csv"), index=False)
    make_boxplots(runs)

    print("\nResults (acceptance: >=15% delay & >=10% CO2, p<0.05):")
    for _, r in stats_df.iterrows():
        print(f"  {r['metric']:<13} improvement={r['improvement_pct']:.1f}% "
              f"(need {r['threshold_pct']:.0f}%)  p={r['p_value']:.3f}  "
              f"r_rb={r['rank_biserial']:.2f}  [{'PASS' if r['passes'] else 'RE-EVALUATE'}]")
    print(f"\nSaved to {RESULTS_DIR}")
    print("Note: for the O4 weight-sensitivity analysis, re-run ga_optimise with "
          "inverted weights (W_DELAY=0.4, W_CO2=0.6) and repeat this evaluation.")
    return stats_df


if __name__ == "__main__":
    main()
