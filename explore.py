#!/usr/bin/env python3
import numpy as np
from gameboy import GBGym

def main():
    # Each element in the dataset has a few properties:
    # state,
    # action,
    # next_state,
    # reward,

    gym = GBGym(step_backwards=True)
    GAME_COUNT = 3
    GAME_LENGTH = 1000

    rewards = np.zeros(44)
    gym.reset()
    dataset = np.zeros(shape=(GAME_COUNT * GAME_LENGTH, 64))

    for game in range(0, GAME_COUNT):
        rewards = np.zeros(44)
        state, _ = gym.reset()
        state = state[0]
        for turn in range(0, GAME_LENGTH):
            previous_state = state
            for action in range(0, 44):
                state, reward, terminated, _ = gym.step(action)
                state = state[0]

                if terminated:
                    rewards[action] = -69
                else:
                    rewards[action] = reward

                gym.step_back()

            if np.all(rewards == -69):
                print('ending game...')
                break

            state_vector = np.concatenate([previous_state], axis=0)
            datapoint = np.concatenate((state_vector, rewards), axis=0)
            best_action = np.argmax(rewards)

            index = game * GAME_LENGTH + turn
            dataset[index, :] = datapoint

            state, reward, _, _ = gym.step(best_action)
            state = state[0]

    with open('dataset.npy', 'wb') as file:
        np.save(file, dataset)

if __name__ == '__main__':
    main()
