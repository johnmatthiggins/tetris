import cv2
import numpy as np


def parse_empty_blocks(rgb_screen):
    bw_screen = cv2.cvtColor(rgb_screen, cv2.COLOR_BGR2GRAY)
    state = np.zeros(10)

    for i in range(10):
        found_empty = False
        for j in range(18):
            x_offset = 16 + i * 8
            y_offset = 143 - j * 8

            if found_empty and bw_screen[y_offset, x_offset] == 0:
                state[i] = 1
                break
            elif bw_screen[y_offset, x_offset] != 0:
                found_empty = True

    return state


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
