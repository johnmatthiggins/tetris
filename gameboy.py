#!/usr/bin/env python3
import pyboy as pb
from gymnasium import Env

import torch
import sys
import cv2
import plotly.express as px
import numpy as np
import pyboy as pb

# local imports
from score import read_score

FPS = 60


class GBGym(Env):
    def __init__(self, device):
        self.device = device
        self.game_over_screen = np.load("game_over.npy")

        self.gameboy = start_gameboy()
        self.sm = self.gameboy.botsupport_manager()

        # different actions possible...
        self.action_space = np.arange(0, 7)
        self.current_score = 0

    def score(self):
        game_score = read_score(self.sm.screen().screen_ndarray())
        return game_score

    # step moves forward two frames...
    def step(self, action):
        gb = self.gameboy
        match action:
            case 0:
                for _ in range(0, 8):
                    gb.tick()
            case 1:
                gb.send_input(pb.WindowEvent.PRESS_ARROW_DOWN)

                for _ in range(7):
                    gb.tick()

                gb.send_input(pb.WindowEvent.RELEASE_ARROW_DOWN)
                gb.tick()
            case 2:
                gb.send_input(pb.WindowEvent.PRESS_ARROW_LEFT)
                for _ in range(7):
                    gb.tick()

                gb.send_input(pb.WindowEvent.RELEASE_ARROW_LEFT)
                gb.tick()
            case 3:
                gb.send_input(pb.WindowEvent.PRESS_ARROW_RIGHT)

                for _ in range(7):
                    gb.tick()

                gb.send_input(pb.WindowEvent.RELEASE_ARROW_RIGHT)
                gb.tick()
            case 4:
                gb.send_input(pb.WindowEvent.PRESS_BUTTON_A)
                gb.tick()
                gb.send_input(pb.WindowEvent.RELEASE_BUTTON_A)
                gb.tick()

                for _ in range(3):
                    gb.tick()
                    gb.tick()

            case 5:
                for _ in range(2):
                    gb.send_input(pb.WindowEvent.PRESS_BUTTON_A)
                    gb.tick()
                    gb.send_input(pb.WindowEvent.RELEASE_BUTTON_A)
                    gb.tick()

                for _ in range(2):
                    gb.tick()
                    gb.tick()
            case 6:
                for _ in range(3):
                    gb.send_input(pb.WindowEvent.PRESS_BUTTON_A)
                    gb.tick()
                    gb.send_input(pb.WindowEvent.RELEASE_BUTTON_A)
                    gb.tick()

                gb.tick()
                gb.tick()

            case _:
                pass

        new_score = self.score()

        # reward is how much the score improved...
        reward = new_score - self.current_score
        self.current_score = new_score

        # get numpy array that represents pixels...
        # chop out all the details other than the board...
        cropped_screen = cv2.cvtColor(self.sm.screen().screen_ndarray(), cv2.COLOR_BGR2GRAY)

        reshaped = np.reshape(
            cropped_screen,
            newshape=(1, cropped_screen.shape[0], cropped_screen.shape[1]),
        )
        observation = torch.tensor(reshaped, device=self.device, dtype=torch.float32)

        truncated = False
        terminated = self.is_game_over()

        if terminated:
            action_reward = 0

        # added extra
        return (observation, reward, terminated, truncated)

    def is_game_over(self):
        rgb_screen = self.sm.screen().screen_ndarray()
        relevant_part = rgb_screen[21 : 68 + 1, 29 : 84 + 1]
        return np.all(relevant_part == self.game_over_screen)

    def close(self):
        self.gameboy.stop()

    def reset(self):
        # just return an empty dict because info isn't being used...
        info = dict()

        f = open("start2.state", "rb")
        self.gameboy.load_state(f)
        f.close()
        cropped_screen = cv2.cvtColor(self.sm.screen().screen_ndarray(), cv2.COLOR_BGR2GRAY)

        state = torch.tensor(
            np.reshape(
                cropped_screen,
                newshape=(1, cropped_screen.shape[0], cropped_screen.shape[1]),
            ),
            device=self.device,
            dtype=torch.float32,
        )

        return (state, info)


# move can be any value from 0 up to and including 13.
# the move is a binary buffer composed of two pieces of information...
# the first 2 bits from starting at the least significant bit are
# reserved for determining the rotational position of the piece.
# 00 => default position (no rotation)
# 01 => single rotation
# 10 => double rotation
# 11 => triple rotation (one rotation the other way)
# The next four bits are reserved for the position of the block horizontally.
# (0000, 1001) <= this doesn't use up the full range of our four bits but that's okay...
# because we only are using 0 to 13 we only need 14 output nodes on our neural network.
def _make_move(gb, move):
    rotations = move & 0x03
    horizontal_position = (move & ~0x03) >> 2

    for _ in range(rotations):
        gb.send_input(pb.WindowEvent.PRESS_BUTTON_A)
        gb.tick()
        gb.send_input(pb.WindowEvent.RELEASE_BUTTON_A)
        gb.tick()
        gb.tick()


def main():
    with start_gameboy() as gb:
        sm = gb.botsupport_manager()
        game_over_screen = np.load("game_over.npy")
        old_game_score = 0

        while not gb.tick():
            wait_n_seconds(gb, 2)
            screen = sm.screen().screen_ndarray()

            # if old_game_score != game_score:
            #     print(game_score)
            #     old_game_score = game_score

            screen = sm.screen().screen_ndarray()[21 : 68 + 1, 29 : 84 + 1]
            is_game_over = np.all(screen == game_over_screen)

            if is_game_over:
                print("GAME IS OVER :(")
                input()


def start_gameboy():
    gb = pb.PyBoy("tetris_dx.sgb")

    gb.set_emulation_speed(target_speed=0)

    f = open("start2.state", "rb")
    gb.load_state(f)
    f.close()

    gb.send_input(pb.WindowEvent.PRESS_BUTTON_START)
    gb.tick()
    gb.send_input(pb.WindowEvent.RELEASE_BUTTON_START)
    gb.tick()

    return gb


def press_start(gb):
    gb.send_input(pb.WindowEvent.PRESS_BUTTON_START)
    gb.tick()

    gb.send_input(pb.WindowEvent.RELEASE_BUTTON_START)
    gb.tick()


def seconds_to_frames(seconds):
    return seconds * FPS


def wait_n_seconds(gb, n):
    i = 0
    while not gb.tick() and i < int(seconds_to_frames(n)):
        i += 1


if __name__ == "__main__":
    main()
