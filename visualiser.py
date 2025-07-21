import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import traci
import time

# ----- Setup -----
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# SUMO configuration
sumo_config = [
    "sumo-gui",
    "-c",
    "SUMO_networks/junction.sumocfg",
    "--step-length",
    "0.05",
    "--delay",
    "1000",
]


# ----- DQN Model -----
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.main(x)


# ----- Environment Setup -----
lane_detectors = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8"]
TRAFFIC_LIGHT_ID = "traffic_light"
DELTA_PHASE_DURATION = 6
YELLOW_PHASE_DURATION = 4

def get_state():
    state = []
    for detector in lane_detectors:
        state.append(traci.lanearea.getLastStepHaltingNumber(detector))
    return torch.tensor(state, dtype=torch.float)

def simulate_time(seconds = 1):
    for i in range(20 * seconds):
        traci.simulationStep()

current_phase = 2

def step(action):
    global current_phase

    if 2 * action == current_phase:
        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, 2 * action)
        simulate_time(DELTA_PHASE_DURATION)
        next_state = get_state()
        next_queue_size = torch.sum(next_state)
        reward =  -next_queue_size
        done = traci.simulation.getMinExpectedNumber() == 0
        return next_state, reward, done, next_queue_size
    else:
        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, current_phase + 1)
        simulate_time(YELLOW_PHASE_DURATION)
        current_phase = 2 * action
        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, 2 * action)
        simulate_time(DELTA_PHASE_DURATION)
        next_state = get_state()
        next_queue_size = torch.sum(next_state)
        reward =  -next_queue_size
        done = traci.simulation.getMinExpectedNumber() == 0
        return next_state, reward, done, next_queue_size

# ----- Load Trained Model -----
state_size = 8
action_size = 8

policy_net = DQN(state_size, action_size)
policy_net.load_state_dict(torch.load("dqn_model.pth"))
policy_net.eval()

# ----- Start SUMO GUI -----
traci.start(sumo_config)
print("Simulation started...")

# ----- Run One Episode -----
state = get_state()
done = False
rewards = []
queue_sizes = []
steps = 0

while not done:
    with torch.no_grad():
        action = torch.argmax(policy_net(state.unsqueeze(0))).item()

    next_state, reward, done, queue_size = step(action)

    rewards.append(reward)
    queue_sizes.append(queue_size)
    state = next_state
    steps += 1

    print(f"Step {steps}: Action={action}, Queue={queue_size:.1f}, Reward={reward:.1f}")

traci.close()

# ----- Plot Results -----
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(queue_sizes)
plt.title("Queue size over time")
plt.xlabel("Time step")
plt.ylabel("Total queue length")

plt.subplot(1, 2, 2)
plt.plot(rewards)
plt.title("Reward over time")
plt.xlabel("Time step")
plt.ylabel("Reward")

plt.tight_layout()
plt.show()
