# WS6 — Experiment pipeline (O2 → O3 → O4)

The development work for Objectives 2–4. All scripts are headless (use `sumo.exe`),
reproducible (fixed seeds), and share one simulation core.

## Files

| File | Objective | What it does |
|---|---|---|
| `sim.py` | (core) | `run_simulation(ns_green, ew_green, seed)` → metrics dict. Installs a 4-phase green/yellow program via TraCI `setProgramLogic`; collects waiting time, CO₂, queue, throughput. |
| `baseline.py` | **O2** | Runs the fixed-time plan (27/27, 60 s cycle) over seeds 42/123/999; computes CV; PASS if CV ≤ 5%. |
| `ga_optimise.py` | **O3** | DEAP GA over `[ns_green, ew_green]`; fitness `F = 0.6·(delay/base) + 0.4·(CO₂/base)`; logs convergence; saves best plan + plot. |
| `evaluate.py` | **O4** | 5 paired seeded runs (GA vs baseline); Wilcoxon signed-rank + rank-biserial; % improvement; box plots. |

## How to run (from `sumo_project/`, using the project venv)

```bash
# venv python: ../sumo_env/Scripts/python.exe
python baseline.py                 # O2  → results/baseline_runs.csv, baseline_cv.csv
python ga_optimise.py --quick      # O3 smoke test (population = 6,generations = 4)
python ga_optimise.py              # O3 full run (population = 20, generations = 50)
python evaluate.py                 # O4  → results/eval_stats.csv, eval_boxplots.png
```

## Key design decisions (traceable to the write-up)

- **Chromosome = `[ns_green, ew_green]`**, greens bounded [10, 60] s, yellows fixed 3 s
  (matches `cross.net.xml` phases 1 & 3). Cycle = ns + ew + 6.
- **Fitness normalises against the O2 baseline** so `F < 1` means "better than baseline";
  the 0.6/0.4 weights are the interpretable, tunable policy lever (WS1/WS3).
- **Training vs evaluation seeds are separated**: the GA optimises on seed 42; O4 re-tests
  the winning plan on seeds 42/123/999/7/2024 (per the evaluation plan §5.4).
- **Weight sweep (WS3)**: to trace the delay–CO₂ trade-off curve and do the O4 sensitivity
  analysis, re-run `ga_optimise.py` with the weights inverted (W_DELAY=0.4, W_CO2=0.6) and
  repeat `evaluate.py`.

## Known consistency notes (align write-up to code)
- Yellow is **3 s** here (net file), not the 5 s mentioned in an early draft; cycle is 60 s.
- Emission model is SUMO's **default HBEFA3** (vType sets no `emissionClass`); the slides say
  "PHEMlight". Pick one and make the draft consistent — to use PHEMlight, set
  `emissionClass="PHEMlight/PC_G_EU4"` on the vType in `cross.rou.xml`.
- Only N→S and E→W flows are defined in `cross.rou.xml`; consider adding the reverse/turning
  flows for a more realistic baseline before the final runs.
