#!/usr/bin/env python
import numpy as np
import torch
import torch.nn.functional as F

import gymnasium as gym
from matris import Game
from matris import create_screen

from state import bumpiness_score

class TetrisGym(gym.Env):
    def __init__(self):
        self.screen = create_screen()

        # left, right, rotate, drop
        self.action_space = torch.arange(4)
        self.game = Game(self.screen)

    def step(self, action):
        action_count = self.action_space.size()[0]
        input_vector = F.one_hot(torch.tensor(action))

        terminated = not self.game.step(input_vector)
        truncated = False
        state = self.game.tensor_state()
        calculated_reward = 0

        reward = 0 if terminated else calculated_reward

        return (state, reward, terminated, truncated)

    def reset(self):
        self.game = Game(self.screen)
        state = self.game.tensor_state()
        info = dict()

        return (state, info)

if __name__ == '__main__':
    tgym = TetrisGym()

    _, _, terminated, _ = tgym.step(np.random.randint(tgym.action_space.size()[0]))
    while not terminated:
        _, _, terminate, _ = tgym.step(np.random.randint(tgym.action_space.size()[0]))
