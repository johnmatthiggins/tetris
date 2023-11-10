#!/usr/bin/env python3
import torch
import numpy as np

from gameboy import GBGym
from tetris import TetrisNN


def main():
    gb = GBGym()
    model = TetrisNN(len(gb.action_space))

    # load weights...
    model.load_state_dict(torch.load('model.pt'))
    model.eval()

    while not gb.is_game_over():
        next_action = model(state).max(1)[1]
        state, reward, terminated, truncated = gb.step(next_action)

    gb.close()


if __name__ == "__main__":
    main()
