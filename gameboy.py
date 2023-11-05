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
    def __init__(self):
        self.game_over_screen = np.load("game_over.npy")

        self.gameboy = start_gameboy()
        self.sm = self.gameboy.botsupport_manager()

        # different actions possible...
        self.action_space = np.arange(0, 6)
        self.current_score = 0

    def score(self):
        game_score = read_score(self.sm.screen().screen_ndarray())
        return game_score

    # step moves forward two frames...
    def step(self, action):
        gb = self.gameboy
        match action:
            case 0:
                gb.send_input(pb.WindowEvent.PRESS_ARROW_UP)
                gb.tick()
                gb.send_input(pb.WindowEvent.RELEASE_ARROW_UP)
            case 1:
                gb.send_input(pb.WindowEvent.PRESS_ARROW_DOWN)
                gb.tick()
                gb.send_input(pb.WindowEvent.RELEASE_ARROW_DOWN)
                gb.tick()
            case 2:
                gb.send_input(pb.WindowEvent.PRESS_ARROW_LEFT)
                gb.tick()
                gb.send_input(pb.WindowEvent.RELEASE_ARROW_LEFT)
                gb.tick()
            case 3:
                gb.send_input(pb.WindowEvent.PRESS_ARROW_RIGHT)
                gb.tick()
                gb.send_input(pb.WindowEvent.RELEASE_ARROW_RIGHT)
                gb.tick()
            case 4:
                gb.send_input(pb.WindowEvent.PRESS_BUTTON_A)
                gb.tick()
                gb.send_input(pb.WindowEvent.RELEASE_BUTTON_A)
                gb.tick()
            case 5:
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
        cropped_screen = self.sm.screen().screen_ndarray()

        observation = torch.from_numpy(
            np.reshape(
                cropped_screen,
                newshape=(3, cropped_screen.shape[0], cropped_screen.shape[1]),
            )
        )

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
        cropped_screen = self.sm.screen().screen_ndarray()

        state = torch.from_numpy(
            np.reshape(
                cropped_screen,
                newshape=(3, cropped_screen.shape[0], cropped_screen.shape[1]),
            )
        )

        return (state, info)


def main():
    with start_gameboy() as gb:
        sm = gb.botsupport_manager()
        game_over_screen = np.load("game_over.npy")
        old_game_score = 0

        while not gb.tick():
            wait_n_seconds(gb, 2)
            screen = sm.screen().screen_ndarray()
            # game_score = read_score(screen)

            fig = px.imshow(screen)
            fig.show()
            input()

            if old_game_score != game_score:
                print(game_score)
                old_game_score = game_score

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
