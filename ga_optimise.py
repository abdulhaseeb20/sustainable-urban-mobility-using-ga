"""O3 - Multi-objective GA over signal green durations (DEAP + TraCI).

Evolves the chromosome ``[ns_green, ew_green]`` to minimise the interpretable
composite fitness

    F = W_DELAY * (delay / baseline_delay) + W_CO2 * (co2 / baseline_co2)

i.e. a weighted sum of each metric normalised against the O2 fixed-time baseline.
F < 1 on a component means the evolved plan beats the baseline on that component.
The weights are the explicit, stakeholder-set policy lever (see WS1/WS3); a weight
sweep over them (evaluate.py / sweep) traces the delay-CO2 trade-off curve.

Outputs (in ./results):
  ga_convergence.csv  - best & mean fitness per generation (evidence artefact)
  ga_convergence.png  - convergence plot
  ga_best.csv         - the best evolved plan and its metrics
"""
import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deap import base, creator, tools

from sim import run_simulation, GREEN_MIN, GREEN_MAX
from baseline import run_baseline, BASELINE_NS_GREEN, BASELINE_EW_GREEN

# Composite-fitness weights (the tunable policy lever).
W_DELAY = 0.6
W_CO2 = 0.4

# Seed used for fitness evaluation during the search ("training" seed).
# O4 re-evaluates the winning plan on a *separate* set of seeds.
TRAIN_SEED = 42

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# DEAP type creation must happen at import time, exactly once.
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

_eval_counter = 0  # unique TraCI label per evaluation


def _clamp(individual):
    for i, gene in enumerate(individual):
        individual[i] = int(min(GREEN_MAX, max(GREEN_MIN, round(gene))))
    return individual


def make_evaluator(base_delay, base_co2, seed=TRAIN_SEED):
    """Return a DEAP evaluate() closure using the given baseline normalisers."""
    def evaluate(individual):
        global _eval_counter
        _eval_counter += 1
        ns, ew = _clamp(individual)
        m = run_simulation(ns, ew, seed=seed, label=f"ga{_eval_counter}")
        norm_delay = m["waiting_time"] / base_delay
        norm_co2 = m["co2"] / base_co2
        fitness = W_DELAY * norm_delay + W_CO2 * norm_co2
        return (fitness,)
    return evaluate


def build_toolbox(evaluate):
    toolbox = base.Toolbox()
    toolbox.register("attr_green", random.randint, GREEN_MIN, GREEN_MAX)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_green, n=2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=6, indpb=0.5)
    toolbox.register("select", tools.selTournament, tournsize=3)
    # Keep genes valid after variation.
    toolbox.decorate("mate", _clamp_decorator)
    toolbox.decorate("mutate", _clamp_decorator)
    return toolbox


def _clamp_decorator(func):
    def wrapper(*args, **kwargs):
        offspring = func(*args, **kwargs)
        for child in offspring:
            _clamp(child)
        return offspring
    return wrapper


def run_ga(pop_size=20, generations=50, cxpb=0.5, mutpb=0.3, rng_seed=1):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    # Baseline normalisers: reuse O2 results if present, else run them.
    runs_csv = os.path.join(RESULTS_DIR, "baseline_runs.csv")
    if os.path.exists(runs_csv):
        base_df = pd.read_csv(runs_csv)
    else:
        base_df = run_baseline()
    base_delay = base_df["waiting_time"].mean()
    base_co2 = base_df["co2"].mean()
    print(f"Baseline normalisers: delay={base_delay:.0f}  co2={base_co2:.0f}")

    toolbox = build_toolbox(make_evaluator(base_delay, base_co2))
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)  # retains the best plan ever seen (elitism)

    # Evaluate the initial population.
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    hof.update(pop)

    history = []

    def log(gen):
        fits = [ind.fitness.values[0] for ind in pop]
        best = hof[0]  # best-so-far, not just this generation's best
        row = {"gen": gen, "best_fitness": best.fitness.values[0],
               "mean_fitness": float(np.mean(fits)),
               "best_ns": best[0], "best_ew": best[1]}
        history.append(row)
        print(f"  gen {gen:>2}: best={row['best_fitness']:.4f}  "
              f"mean={row['mean_fitness']:.4f}  plan=[{best[0]},{best[1]}]")

    log(0)
    for gen in range(1, generations + 1):
        # Reserve one slot for the elite carried over from hof.
        offspring = toolbox.select(pop, len(pop) - 1)
        offspring = [toolbox.clone(i) for i in offspring]
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values
        for m in offspring:
            if random.random() < mutpb:
                toolbox.mutate(m)
                del m.fitness.values
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)
        offspring.append(toolbox.clone(hof[0]))  # elitism: carry best forward
        pop[:] = offspring
        hof.update(pop)
        log(gen)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(RESULTS_DIR, "ga_convergence.csv"), index=False)

    # Convergence plot.
    plt.figure(figsize=(7, 4))
    plt.plot(hist_df["gen"], hist_df["best_fitness"], label="best", marker="o", ms=3)
    plt.plot(hist_df["gen"], hist_df["mean_fitness"], label="mean", alpha=0.7)
    plt.axhline(1.0, color="grey", ls="--", lw=1, label="baseline (F=1)")
    plt.xlabel("generation"); plt.ylabel("composite fitness F")
    plt.title("GA convergence"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ga_convergence.png"), dpi=120)

    best = hof[0]
    best_metrics = run_simulation(best[0], best[1], seed=TRAIN_SEED, label="ga_best")
    best_metrics["fitness"] = best.fitness.values[0]
    pd.DataFrame([best_metrics]).to_csv(
        os.path.join(RESULTS_DIR, "ga_best.csv"), index=False)

    # O3 acceptance: no regression + at least one plan beats baseline.
    no_regression = hist_df["best_fitness"].iloc[-1] <= hist_df["best_fitness"].iloc[0]
    beats_baseline = hist_df["best_fitness"].min() < 1.0
    print(f"\nBest plan: NS={best[0]}s EW={best[1]}s  F={best.fitness.values[0]:.4f}")
    print(f"O3 acceptance: no_regression={no_regression}  beats_baseline={beats_baseline}")
    print(f"  -> {'PASS' if (no_regression and beats_baseline) else 'REVIEW'}")
    return hist_df, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--gens", type=int, default=50)
    ap.add_argument("--quick", action="store_true",
                    help="tiny run (pop 6, 4 gens) to validate the pipeline")
    args = ap.parse_args()
    if args.quick:
        run_ga(pop_size=6, generations=4)
    else:
        run_ga(pop_size=args.pop, generations=args.gens)


if __name__ == "__main__":
    main()
