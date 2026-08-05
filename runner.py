import os
import sys
import time 

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("ERROR: Please declare environment variable 'SUMO_HOME'")

import traci

sumo_binary = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"
sumo_cmd = [sumo_binary, "-c", "cross.sumocfg", "--start"]

def run_simulation():
    print("Starting TraCI and connecting to SUMO...")
    traci.start(sumo_cmd)
    
    step = 0
    incoming_edges = ["top_to_center", "bottom_to_center", "left_to_center", "right_to_center"]
    
    print("Python has successfully taken control of the Traffic Lights!")
    
    while step < 1000:
        traci.simulationStep() 
        
        # NEW: pause for 0.1 seconds (100ms) every step ---
        time.sleep(0.1) 
        
        cycle_second = step % 60 
        
        if cycle_second == 0:
            traci.trafficlight.setPhase("center", 0) 
        elif cycle_second == 25:
            traci.trafficlight.setPhase("center", 1) 
        elif cycle_second == 30:
            traci.trafficlight.setPhase("center", 2) 
        elif cycle_second == 55:
            traci.trafficlight.setPhase("center", 3) 
            
        waiting_cars = 0
        for edge in incoming_edges:
            waiting_cars += traci.edge.getLastStepHaltingNumber(edge)
            
        if step % 60 == 0:
            print(f"Cycle completed at Step {step} | Total waiting cars: {waiting_cars}")
            
        step += 1

    print("Simulation complete. Closing TraCI.")
    traci.close()

if __name__ == "__main__":
    run_simulation()