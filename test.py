# Step 1: Add modules to provide access to specific libraries and functions
import os # Module provides functions to handle file paths, directories, environment variables
import sys # Module provides access to Python-specific system parameters and functions

# Step 2: Establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Step 3: Add Traci module to provide access to specific libraries and functions
import traci # Module for controlling SUMO simulations via TraCI

# Step 4: Define SUMO configuration
Sumo_config = [
    'sumo-gui',
    '-c', 'SUMO_networks/junction.sumocfg',
    '--step-length', '0.05',
    '--delay', '1000',
    '--lateral-resolution', '0.1'
]

# Step 5: Open connection between SUMO and Traci
traci.start(Sumo_config)

# Step 6: Define Variables
lane_detectors = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8']

# Step 7: Define Functions
def print_queue_lengths():
    print("Queue status:")
    for det_id in lane_detectors:
        halted = traci.lanearea.getLastStepHaltingNumber(det_id)
        print(f"  {det_id}: {halted} vehicles in queue")


def get_vehicle_speeds():
    for veh_id in traci.vehicle.getIDList():
        speed = traci.vehicle.getSpeed(veh_id)
        print(f"{veh_id}: {speed:.3f} m/s")

# Step 8: Take simulation steps until there are no more vehicles in the network
num_steps = 0
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    num_steps += 1
    if num_steps % 20 == 0:  # every 1s
        print(f"\nTime: {num_steps/20:.2f}s")
        print_queue_lengths()
        program = traci.trafficlight.getAllProgramLogics("traffic_light")[0]
        print(program)
        print("------------")
        print(program.phases[0])
        print(program.phases[1])
# Step 9: Close connection between SUMO and Traci
traci.close()
