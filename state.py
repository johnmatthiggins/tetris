import torch
import cv2
import numpy as np
import plotly.express as px

# local imports
from piece import erase_piece

def bumpiness_score(block_map):
    screen = block_map[2:, :]
    bump_vector = np.zeros(block_map.shape[1])
    bumpiness_score = 0

    for i in range(screen.shape[1]):
        if np.any(screen[:, i] > 0):
            bump_vector[i] = screen.shape[0] - np.argmax(screen[:, i] > 0)

    for i in range(1, screen.shape[1]):
        diff = np.abs(bump_vector[i] - bump_vector[i - 1])
        bumpiness_score += diff

    print(bump_vector)
    print('bumpiness: %s' % str(bumpiness_score))
    return bumpiness_score

def build_block_map(rgb_screen):
    bw_screen = cv2.cvtColor(rgb_screen, cv2.COLOR_BGR2GRAY)
    state = np.zeros(shape=(18, 10))

    for i in range(10):
        found_empty = False
        for j in range(18):
            x_offset = 16 + i * 8
            y_offset = 143 - j * 8

            space_filled = bw_screen[y_offset, x_offset] == 0
            state[j, i] = 1 if space_filled else 0

    return np.flip(state, axis=0)


def find_empty_blocks(block_map):
    block_map = block_map[2:, :]
    state = torch.zeros((10,))

    for i in range(block_map.shape[1]):
        strip = block_map[:, i]
        if np.any(strip > 0):
            end = np.argmax(strip > 0)
            j = 0
            while j < (strip.shape[0] - end - 1):
                y_offset = strip.shape[0] - j - 1

                if strip[y_offset] > 0:
                    state[i] += 1

                j += 1

    return state
