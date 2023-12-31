#!/usr/bin/env python
import numpy as np
import torch
import torch.nn.functional as F

import gymnasium as gym
from matris.game import Game
from matris.game import create_screen

from state import bumpiness_score
from state import find_empty_blocks

from matris.tetrominoes import tetrominoes

tetromino_indices = {
    "blue": 0,
    "yellow": 1,
    "pink": 2,
    "green": 3,
    "red": 4,
    "cyan": 5,
    "orange": 6,
}

class TetrisGym(gym.Env):
    def __init__(self, device="mps"):
        self.screen = create_screen()

        # left, right, rotate, drop
        self.action_space = torch.arange(4)
        self.game = Game(self.screen)
        self.device = device

        self.score = 0

    def step(self, action):
        action_count = self.action_space.size()[0]
        input_vector = F.one_hot(torch.tensor(action))

        terminated = not self.game.step(input_vector)
        truncated = False
        state = self.game.tensor_state()
        _, bump_vector = bumpiness_score(state)

        observation = torch.cat([self._piece_vector(), bump_vector]).unsqueeze(0)
        observation = torch.tensor(observation, device=self.device)

        new_score = self._score(state)
        calculated_reward = new_score - self.score

        self.score = new_score

        reward = 0 if terminated else calculated_reward

        return (observation, reward, terminated, truncated)

    def reset(self):
        self.game = Game(self.screen)
        state = self.game.tensor_state()
        _, bump_vector = bumpiness_score(state)
        observation = torch.cat([self._piece_vector(), bump_vector]).unsqueeze(0)
        observation = torch.tensor(observation, device=self.device)
        info = dict()

        return (observation, info)

    def _piece_vector(self):
        current = tetromino_indices[self.game.matris.next_tetromino.color]
        current = tetromino_indices[self.game.matris.current_tetromino.color]
        rotation = self.game.matris.tetromino_rotation
        x_position, y_position = self.game.matris.tetromino_position

        # pad with 5 zeros...
        return torch.tensor([
            current,
            rotation,
            x_position,
            y_position,
            0, 0, 0, 0, 0, 0,
        ])
    
    def _score(self, state):
        line_score = self.game.lines()
        point_score = self.game.lines()

        bumpiness, bump_vector = bumpiness_score(state)
        height = torch.sum(bump_vector)
        empty_blocks = find_empty_blocks(state).sum()

        new_aggregated_score = (point_score - 0.51 * height + line_score ** 2
                                - 10 * empty_blocks - 0.18 * bumpiness)

        return new_aggregated_score
        

if __name__ == '__main__':
    tgym = TetrisGym()

    _, _, terminated, _ = tgym.step(np.random.randint(tgym.action_space.size()[0]))

    while not terminated:
        _, _, terminate, _ = tgym.step(np.random.randint(tgym.action_space.size()[0]))

