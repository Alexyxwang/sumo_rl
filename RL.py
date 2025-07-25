import os
import sys
import random
from collections import deque
import torch.optim as optim

import math
import time

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# ------------------------------------------------------------------

import traci

sumo_config = [
    "sumo",
    "-c", "SUMO_networks/junction.sumocfg",
    "--step-length", "0.05",
    "--delay", "0",
    "--lateral-resolution", "0.1",
    "--start",
    "--no-warnings",
    "--no-step-log",
]

traci.start(sumo_config)

TRAFFIC_LIGHT_ID = "traffic_light"
DELTA_PHASE_DURATION = 6
YELLOW_PHASE_DURATION = 4
lane_detectors = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8']
action = []


def change_env():
    traci.close()
    traci.start(sumo_config)

def get_queue_length():
    state = []
    for detector in lane_detectors:
        state.append(traci.lanearea.getLastStepHaltingNumber(detector))
    # print(state)
    # print(torch.tensor(state, dtype=torch.float))
    return torch.tensor(state, dtype=torch.float)

def get_current_state():
    return get_queue_length()

def simulate_time(seconds = 1):
    for _ in range(20 * seconds):
        traci.simulationStep()

current_phase = 2

def step(action):
    global current_phase

    if 2 * action == current_phase:
        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, 2 * action)
        simulate_time(DELTA_PHASE_DURATION)
        next_state = get_current_state()
        next_queue_size = torch.sum(next_state)
        reward =  -next_queue_size
        done = traci.simulation.getMinExpectedNumber() == 0
        return next_state, reward, done
    else:
        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, current_phase + 1)
        simulate_time(YELLOW_PHASE_DURATION)
        current_phase = 2 * action
        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, 2 * action)
        simulate_time(DELTA_PHASE_DURATION)
        next_state = get_current_state()
        next_queue_size = torch.sum(next_state)
        reward =  -next_queue_size
        done = traci.simulation.getMinExpectedNumber() == 0
        return next_state, reward, done


# ------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, state_space_size, action_space_size):
        super(DQN, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(state_space_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_space_size),
        )

    def forward(self, x):
        output = self.main(x)
        return output


learning_rate = 0.01

# discount factor
gamma = 0.999

# starting exploration rate
epsilon = 0.9

# rate of decay as model becomes more stable (5%)
epsilon_decay = 0.95

# final exploration rate for stable model
min_epsilon = 0.05
batch_size = 128
target_update_freq = 400
memory_size = 10000
episodes = 5
# episodes = 75
# episodes = 200

state = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float)
action_space_size = 8

policy_net = DQN(state.shape[0], action_space_size)
target_net = DQN(state.shape[0], action_space_size)

# Ensure both neural networks have the same initial parameters
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = deque(maxlen=memory_size)


def choose_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, action_space_size - 1)
    else:
        q_values = policy_net(state.unsqueeze(0))
        return torch.argmax(q_values).item()


def optimise_model():
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)

    state_batch = torch.stack([b[0] for b in batch]).float()
    next_state_batch = torch.stack([b[3] for b in batch]).float()
    reward_batch = torch.tensor([b[2] for b in batch], dtype=torch.float)
    action_batch = torch.tensor([b[1] for b in batch], dtype=torch.long).unsqueeze(1)
    done_batch = torch.tensor([b[4] for b in batch], dtype=torch.float)

    q_values = policy_net(state_batch).gather(1, action_batch).squeeze()

    with torch.no_grad():
        max_next_q_values = target_net(next_state_batch).max(1)[0]
        target_q_values = reward_batch + gamma * max_next_q_values * (1 - done_batch)

    loss = nn.MSELoss()(q_values, target_q_values)

    # print(f"Training loss: {loss.item()}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# ------------------------------------------------------------------

rewards_per_episode = []
steps_done = 0

# ADDITIONS
avg_wait_per_ep = []
max_wait_per_ep = []

num_lanes = len(lane_detectors)
all_avg_queue_lengths = torch.zeros(num_lanes, episodes)


for episode in range(episodes):
    print(f"Episode {episode}")
    change_env()

    state = get_current_state()
    episode_reward = 0
    done = False

    vehicle_wait_tracker = {} # NEW
    queue_length_tracker = {}
    num_steps = 0
    while not done:
        num_steps += 1 
        # Select action
        action = choose_action(state, epsilon)
        next_state, reward, done = step(action)

        # ADDITIONS
        for v_id in traci.vehicle.getIDList():
            wait_time = traci.vehicle.getWaitingTime(v_id)

            if v_id not in vehicle_wait_tracker:
                vehicle_wait_tracker[v_id] = wait_time
            elif wait_time > vehicle_wait_tracker[v_id]:
                vehicle_wait_tracker[v_id] = wait_time
        #

        # retrive queue length: list of size 8 (one number for each lane detector)
        curr_queue = get_queue_length()
        for i in range(len(curr_queue)):
            if i not in queue_length_tracker:
                queue_length_tracker[i] = curr_queue[i]
            else:
                queue_length_tracker[i] += curr_queue[i]

        # sum each lane's queue length for every step (lane by lane)
        # at the end of the episode, find average lane queue length per lane


        # print(f"Action={action}, Reward={reward:.2f}, Done={done}")
        # print(f"Next State: {next_state.tolist()}")
        # Store transition in memory
        memory.append((state, action, reward, next_state, done))

        # Update state
        state = next_state
        episode_reward += reward
        # print(episode_reward)
        # Optimize model
        optimise_model()

        # Update target network periodically
        if steps_done % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        steps_done += 1

    # ADDITIONS
    
    for i, len_i in queue_length_tracker.items():
        all_avg_queue_lengths[i, episode] = len_i / num_steps
        
    avg_wait = 0.0
    max_wait = 0.0
    vehicle_waits = []

    for key in vehicle_wait_tracker:
        vehicle_waits.append(vehicle_wait_tracker[key])

    if vehicle_waits:
        avg_wait = sum(vehicle_waits) / len(vehicle_waits)
        max_wait = max(vehicle_waits)

    avg_wait_per_ep.append(avg_wait)
    max_wait_per_ep.append(max_wait)
    # 
    
    # Decay epsilon
    epsilon = max(min_epsilon, epsilon_decay * epsilon)
    print(episode_reward)
    rewards_per_episode.append(episode_reward)

torch.save(policy_net.state_dict(), "dqn_model.pth")

import matplotlib.pyplot as plt

# Total reward per episode
plt.figure()
plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN on traffic lights")
plt.grid(True)
plt.tight_layout()
plt.savefig("Plots/rewards_per_episode.png")
plt.close()

# Average wait time per episode
plt.figure()
plt.plot(avg_wait_per_ep)
plt.xlabel("Episode")
plt.ylabel("Average Wait")
plt.title("DQN on traffic lights - Avg Wait")
plt.grid(True)
plt.tight_layout()
plt.savefig("Plots/avg_wait_per_episode.png")
plt.close()

# Maximum wait time per episode
plt.figure()
plt.plot(max_wait_per_ep)
plt.xlabel("Episode")
plt.ylabel("Maximum Wait")
plt.title("DQN on traffic lights - Max Wait")
plt.grid(True)
plt.tight_layout()
plt.savefig("Plots/max_wait_per_episode.png")
plt.close()

# Per-lane average queue length
num_lanes = all_avg_queue_lengths.shape[0]
for lane_index in range(num_lanes):
    plt.figure()
    plt.plot(all_avg_queue_lengths[lane_index].numpy())
    plt.xlabel("Episode")
    plt.ylabel("Avg Queue Length")
    plt.title(f"Avg queue length per episode for lane {lane_index}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Plots/avg_queue_lane_{lane_index}.png")
    plt.close()
