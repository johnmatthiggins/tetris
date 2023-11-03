#!/usr/bin/env python3
from game import GBGym
import numpy as np

def main():
    gb = GBGym()

    while not gb.is_game_over():
        next_action = np.random.choice(gb.action_space)
        gb.step(next_action)

    gb.close()

if __name__ == '__main__':
    main()
