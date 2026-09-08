"""O5 - Spatial Heatmap Visualization.

Reads the parameters and metrics from baseline_runs.csv and ga_best.csv,
runs a single visualization replay to extract spatial edge data, 
and plots a geographic heatmap of intersection congestion.
"""
import os
import pandas as pd
import sumolib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import xml.etree.ElementTree as ET

from sim import run_simulation

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
NET_FILE = "cross.net.xml"  # UPDATE THIS to match your CONFIG's network file

def create_dump_config(config_name, output_name):
    with open(config_name, "w") as f:
        f.write(f'<additional><edgeData id="heatmap" file="{output_name}" freq="3600" excludeEmpty="true"/></additional>')

def extract_delays(xml_file):
    edge_delays = {}
    if not os.path.exists(xml_file):
        return edge_delays
    tree = ET.parse(xml_file)
    for interval in tree.getroot().findall('interval'):
        for edge in interval.findall('edge'):
            edge_id = edge.get('id')
            delay = float(edge.get('waitingTime', 0))
            edge_delays[edge_id] = edge_delays.get(edge_id, 0) + delay
    return edge_delays

def plot_heatmap(ax, net, edge_delays_dict, title, subtitle, norm, cmap, global_max):
    ax.set_facecolor('#1a1a1a')
    ax.set_title(title, color='white', size=16, pad=25)
    ax.text(0.5, 1.02, subtitle, color='#aaaaaa', size=12, ha='center', transform=ax.transAxes)
    
    for edge in net.getEdges():
        edge_id = edge.getID()
        delay = edge_delays_dict.get(edge_id, 0)
        
        color = cmap(norm(delay)) if delay > 0 else '#333333'
        line_weight = (delay / global_max * 6) + 1 if global_max > 0 else 2
        
        shape = edge.getShape()
        x, y = [pt[0] for pt in shape], [pt[1] for pt in shape]
        ax.plot(x, y, color=color, linewidth=line_weight, solid_capstyle='round')
    ax.axis('off')

def main():
    print("1. Reading data from CSV results...")
    base_csv = os.path.join(RESULTS_DIR, "baseline_runs.csv")
    ga_csv = os.path.join(RESULTS_DIR, "ga_best.csv")

    if not os.path.exists(base_csv) or not os.path.exists(ga_csv):
        print("Error: Missing CSV results. Run baseline.py and ga_optimization.py first.")
        return

    # Extract Baseline data (using mean of all seeds for the text display, but seed 42 timings)
    base_df = pd.read_csv(base_csv)
    base_ns = int(base_df["ns_green"].iloc[0])
    base_ew = int(base_df["ew_green"].iloc[0])
    base_delay_mean = base_df["waiting_time"].mean()

    # Extract GA data
    ga_df = pd.read_csv(ga_csv)
    ga_ns = int(ga_df["ns_green"].iloc[0])
    ga_ew = int(ga_df["ew_green"].iloc[0])
    ga_delay = float(ga_df["waiting_time"].iloc[0])
    
    # Calculate improvement
    delay_diff = ((base_delay_mean - ga_delay) / base_delay_mean) * 100

    print(f" -> Baseline : NS={base_ns}, EW={base_ew} | Delay: {base_delay_mean:.0f}")
    print(f" -> GA Best  : NS={ga_ns}, EW={ga_ew} | Delay: {ga_delay:.0f} ({delay_diff:.1f}% improvement)")

    print("\n2. Generating XML configurations...")
    create_dump_config("dump_baseline.xml", "baseline_data.xml")
    create_dump_config("dump_ga.xml", "ga_best_data.xml")

    print("3. Replaying simulations to extract spatial data...")
   
    run_simulation(base_ns, base_ew, seed=42, label="heat_base", dump_file="dump_baseline.xml")
    run_simulation(ga_ns, ga_ew, seed=42, label="heat_ga", dump_file="dump_ga.xml")

    print("\n4. Rendering Heatmaps...")
    net = sumolib.net.readNet(NET_FILE)
    baseline_delays = extract_delays('baseline_data.xml')
    ga_delays = extract_delays('ga_best_data.xml')

    max_base = max(baseline_delays.values()) if baseline_delays else 0
    max_ga = max(ga_delays.values()) if ga_delays else 0
    global_max = max(max_base, max_ga, 1)

    norm = mcolors.Normalize(vmin=0, vmax=global_max)
    cmap = plt.colormaps.get_cmap('YlOrRd')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor('#1a1a1a')

    # Plot Baseline
    title1 = f"Baseline Plan [NS: {base_ns}s, EW: {base_ew}s]"
    sub1 = f"Total Network Delay: {base_delay_mean:.0f}s"
    plot_heatmap(ax1, net, baseline_delays, title1, sub1, norm, cmap, global_max)

    # Plot GA
    title2 = f"GA Optimized Plan [NS: {ga_ns}s, EW: {ga_ew}s]"
    sub2 = f"Total Network Delay: {ga_delay:.0f}s (↓ {delay_diff:.1f}%)"
    plot_heatmap(ax2, net, ga_delays, title2, sub2, norm, cmap, global_max)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='horizontal', fraction=0.05, pad=0.05, aspect=50)
    cbar.set_label('Street-Level Waiting Time (seconds)', color='white', size=12)
    cbar.ax.xaxis.set_tick_params(color='white', labelcolor='white')

    for f in ["dump_baseline.xml", "dump_ga.xml", "baseline_data.xml", "ga_best_data.xml"]:
        if os.path.exists(f): os.remove(f)

    plt.show()

if __name__ == "__main__":
    main()