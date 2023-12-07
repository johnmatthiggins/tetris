#!/usr/bin/env python3
import numpy as np
from gameboy import GBGym

def main():
    gym = GBGym(step_backwards=True)

    rewards = np.zeros(44)
    gym.reset()

    while True:
        for action in range(0, 44):
            _, reward, terminated, _ = gym.step(action)

            if terminated:
                rewards[action] = -69
            else:
                rewards[action] = reward

            gym.step_back()

        if np.all(rewards == -69):
            print('ending game...')
            break

        best_action = np.argmax(rewards)
        gym.step(best_action)
        print(rewards)

if __name__ == '__main__':
    main()
