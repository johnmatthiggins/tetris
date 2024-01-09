import torch
import cv2
import numpy as np
import plotly.express as px

def bumpiness_score(block_map):
    screen = torch.tensor(block_map[2:, :], device="mps")
    bump_vector = torch.zeros(block_map.shape[1], device="mps")
    bumpiness_score = 0

    for i in range(screen.shape[1]):
        if torch.any(screen[:, i] > 0):
            bump_vector[i] = screen.shape[0] - torch.argmax(screen[:, i] > 0)

    for i in range(1, screen.shape[1]):
        diff = torch.abs(bump_vector[i] - bump_vector[i - 1])
        bumpiness_score += diff

    return bumpiness_score, bump_vector


def build_block_map(rgb_screen, tensor=False):
    bw_screen = cv2.cvtColor(rgb_screen, cv2.COLOR_BGR2GRAY)
    if tensor:
        state = torch.zeros(size=(18, 10))
    else:
        state = np.zeros(shape=(18, 10))

    for i in range(10):
        found_empty = False
        for j in range(18):
            x_offset = 16 + i * 8
            y_offset = 143 - j * 8

            space_filled = bw_screen[y_offset, x_offset] == 0
            state[j, i] = 1 if space_filled else 0

    if tensor:
        result = torch.flip(state, dims=(0,))
    else:
        result = np.flip(state, axis=0)

    return result


def find_empty_blocks(block_map):
    block_map = torch.flip(block_map[2:, :], dims=(0,))
    state = torch.zeros((10,), device='mps')

    for i in range(block_map.shape[1]):
        strip = block_map[:, i]
        if torch.any(strip > 0):
            end = strip.shape[0] - torch.argmax((torch.flip(strip, dims=(0,)) > 0).to(torch.long)).item() - 1
            j = 0
            while j < end:
                if strip[j] == 0:
                    state[i] += 1

                j += 1
    return state
