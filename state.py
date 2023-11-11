import torch
import cv2
import numpy as np
import plotly.express as px

# local imports
from piece import erase_piece

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

    return torch.tensor(state)


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


def find_empty_blocks(block_map, piece_state):
    state = torch.zeros((10,))

    no_piece = erase_piece(
            block_map=np.copy(block_map),
            rotation=piece_state.rotation_position,
            piece=piece_state.current_piece,
            position=(piece_state.x, piece_state.y)
        )

    for i in range(10):
        found_empty = False
        for j in range(18):
            x_offset = i
            y_offset = no_piece.shape[0] - j - 1
            
            # if the bottom block is empty, then anything above it is floating.
            if j == 0 and no_piece[y_offset, x_offset] == 0:
                found_empty = True
            elif found_empty and no_piece[y_offset, x_offset] == 1:
                state[i] += 1
            elif no_piece[y_offset, x_offset] == 0 and no_piece[y_offset + 1, x_offset] == 1:
                state[i] += 1
                found_empty = True

    return state

