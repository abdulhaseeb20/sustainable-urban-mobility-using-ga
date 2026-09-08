"""Reusable SUMO simulation runner for the traffic-signal GA project.

Runs one signal-controlled simulation for a given plan (chromosome
``[ns_green, ew_green]``) and returns aggregate performance metrics. Shared by:
  * baseline.py     (O2 - fixed-time baseline + CV stability)
  * ga_optimise.py  (O3 - GA search over green durations)
  * evaluate.py     (O4 - paired GA-vs-baseline evaluation)

The signal is controlled by installing a 4-phase program via ``setProgramLogic``
(green-yellow-green-yellow), whose green durations come from the chromosome. This
is more robust than per-step ``setPhase`` calls because SUMO then runs the program
itself with exactly the durations we specify.

Phase order matches ``cross.net.xml`` tlLogic "center":
  0 = NS green, 1 = NS yellow (3 s), 2 = EW green, 3 = EW yellow (3 s).
"""
import os
import sys

if "SUMO_HOME" not in os.environ:
    sys.exit("ERROR: environment variable 'SUMO_HOME' is not set.")
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci  # noqa: E402  (import must follow SUMO_HOME/tools on sys.path)

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(_HERE, "cross.sumocfg")
INCOMING_EDGES = ["top_to_center", "bottom_to_center", "left_to_center", "right_to_center"]

YELLOW = 3  # seconds; fixed, matches net phases 1 & 3

# Phase state strings copied verbatim from cross.net.xml tlLogic "center".
_PHASE_STATES = [
    "GGGggrrrrrGGGggrrrrr",  # 0 NS green
    "yyyyyrrrrryyyyyrrrrr",  # 1 NS yellow
    "rrrrrGGGggrrrrrGGGgg",  # 2 EW green
    "rrrrryyyyyrrrrryyyyy",  # 3 EW yellow
]

# Sensible bounds for green durations (seconds) - used by the GA too.
GREEN_MIN = 10
GREEN_MAX = 60


def _sumo_binary(gui):
    name = "sumo-gui.exe" if gui else "sumo.exe"
    return os.path.join(os.environ["SUMO_HOME"], "bin", name)

# can be run with GUI: just add a parameter gui=True to run with GUI
def run_simulation(ns_green, ew_green, seed=42, gui=False, max_steps=3600,
                   label="default", dump_file=None):
    """Run one simulation and return a metrics dict.

    Returns
    -------
    dict with keys:
      ns_green, ew_green, cycle, seed  - the plan that was run
      waiting_time  - total vehicle waiting-time (vehicle-seconds), incoming edges
      co2           - total CO2 emitted (mg), incoming edges
      mean_queue    - mean halting-vehicle count per step, incoming edges
      throughput    - number of vehicles that completed their route
      steps         - simulation steps executed
    """
    ns_green = int(round(ns_green))
    ew_green = int(round(ew_green))
    cycle = ns_green + ew_green + 2 * YELLOW

    cmd = [
        _sumo_binary(gui), "-c", CONFIG,
        "--seed", str(seed),
        "--no-step-log", "true",
        "--no-warnings", "true",
        "--duration-log.disable", "true",
    ]
    if dump_file:
        cmd.extend(["-a", dump_file])
    if gui:
        cmd.append("--start")

    traci.start(cmd, label=label)
    conn = traci.getConnection(label)

    # Install our green/yellow program derived from the chromosome.
    phases = [
        traci.trafficlight.Phase(ns_green, _PHASE_STATES[0]),
        traci.trafficlight.Phase(YELLOW, _PHASE_STATES[1]),
        traci.trafficlight.Phase(ew_green, _PHASE_STATES[2]),
        traci.trafficlight.Phase(YELLOW, _PHASE_STATES[3]),
    ]
    logic = traci.trafficlight.Logic("ga_prog", 0, 0, phases)
    conn.trafficlight.setProgramLogic("center", logic)

    total_wait = 0.0
    total_co2 = 0.0
    total_queue = 0
    throughput = 0
    step = 0
    try:
        while step < max_steps and conn.simulation.getMinExpectedNumber() > 0:
            conn.simulationStep()
            for edge in INCOMING_EDGES:
                total_wait += conn.edge.getWaitingTime(edge)
                total_co2 += conn.edge.getCO2Emission(edge)
                total_queue += conn.edge.getLastStepHaltingNumber(edge)
            throughput += conn.simulation.getArrivedNumber()
            step += 1
    finally:
        conn.close()

    return {
        "ns_green": ns_green,
        "ew_green": ew_green,
        "cycle": cycle,
        "seed": seed,
        "waiting_time": total_wait,
        "co2": total_co2,
        "mean_queue": total_queue / step if step else 0.0,
        "throughput": throughput,
        "steps": step,
    }


if __name__ == "__main__":
    # Quick smoke test: one short headless run.
    m = run_simulation(27, 27, seed=42, max_steps=600, label="smoke")
    print("Smoke run:", m)
