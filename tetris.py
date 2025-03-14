#!/usr/bin/env python3
import time
from collections import namedtuple, deque
from itertools import count
import sys
import math
import random

import plotly.express as px
import numpy as np
import pandas as pd
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# local imports
from gameboy import GBGym
from state import bumpiness_score

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

def main():
    live_feed = "--live-feed" in sys.argv

    episode_scores = list()

    env = GBGym(device=DEVICE)
    n_actions = env.action_space.shape[0]

    torch.device(DEVICE)

    # Get the number of state observations
    state, info = env.reset()

    policy_net = TetrisNN(n_actions).to(DEVICE)

    if "--load" in sys.argv:
        print("LOADING PRE-TRAINED MODEL")
        policy_net.load_state_dict(torch.load("model.pt"))
        policy_net.eval()

    target_net = TetrisNN(n_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.RMSprop(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_SIZE)

    num_episodes = 3000

    for i_episode in range(num_episodes):
        episode_score = 0
        start = time.time()

        # Initialize the environment and get it's state
        state, info = env.reset()
        state = state.unsqueeze(0)
        print("Starting episode... [%d/%d]" % (i_episode + 1, num_episodes))

        for t in count():
            action = select_action(policy_net, env, state)
            observation, reward, terminated, truncated = env.step(action.item())
            episode_score += reward

            reward = torch.tensor([reward], device=DEVICE)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = observation.unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(policy_net, target_net, memory, optimizer)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()

            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_scores.append(episode_score)
                break
        end = time.time()
        duration_s = end - start
        print("Episode took %s seconds..." % str(duration_s))

        if i_episode % 1000 == 0:
            torch.save(policy_net.state_dict(), "model_tmp.pt")

    torch.save(policy_net.state_dict(), "model.pt")

    plot_durations(episode_scores, show_result=True)


def get_device():
    device = "cpu"
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print(
                "MPS not available because the current PyTorch install was not "
                "built with MPS enabled."
            )
        else:
            print(
                "MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine."
            )

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    return device


DEVICE = get_device()


class ReplayMemory(object):
    def __init__(self, capacity):
        print("MEMORY_LENGTH = %d" % capacity)
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class TetrisNN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.device = get_device()

        self.conv1 = nn.Conv1d(1, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 3)
        self.conv4 = nn.Conv1d(128, 128, 3)
        self.conv5 = nn.Conv1d(128, 128, 3)
        self.conv6 = nn.Conv1d(128, 128, 3)
        self.conv7 = nn.Conv1d(128, 128, 3)
        self.conv8 = nn.Conv1d(128, 64, 3)

        self.fc1 = nn.Linear(452, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 128)
        self.fc7 = nn.Linear(128, 128)
        self.fc8 = nn.Linear(128, 128)
        self.fc9 = nn.Linear(128, 128)
        self.fc10 = nn.Linear(128, 128)
        self.fc11 = nn.Linear(128, 64)
        self.fc12 = nn.Linear(64, 64)
        self.fc13 = nn.Linear(64, 64)
        self.fc14 = nn.Linear(64, n_actions)

    def forward(self, x):
        piece_indexes = torch.arange(start=0, end=4, dtype=torch.long)

        # extract piece state vector
        piece_state = x[:, 0, piece_indexes]

        # slice away piece vector...
        screen_range = torch.arange(start=1, end=x.shape[2], dtype=torch.long)
        x = x[:, :, screen_range]
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))

        x = torch.flatten(x, 1)
        x = torch.cat([x, piece_state], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = F.relu(self.fc10(x))
        x = F.relu(self.fc11(x))
        x = F.relu(self.fc12(x))
        x = F.relu(self.fc13(x))

        return self.fc14(x)


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``Adam`` optimizer
BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-5

MEMORY_SIZE = 2048

def select_action(policy_net, env, state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            result = policy_net(state).max(1)[1]
            return result.view(1, 1)
    else:
        return torch.tensor(
            [[np.random.choice(env.action_space)]], device=DEVICE, dtype=torch.long
        )


def plot_durations(episode_scores, show_result=False):
    durations_t = torch.tensor(episode_scores, dtype=torch.float)
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))

    df = pd.DataFrame.from_dict(
        {
            "episode_score": durations_t.numpy(),
            "episode_index": np.arange(0, len(durations_t)),
        }
    )

    fig = px.line(df, x="episode_index", y="episode_score")
    fig.show()


def optimize_model(policy_net, target_net, memory, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=DEVICE,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)

    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


steps_done = 0


if __name__ == "__main__":
    main()
