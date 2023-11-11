#!/usr/bin/env python3
from dataclasses import dataclass
import time
import sys

import pyboy as pb
from gymnasium import Env

import torch
import cv2
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pyboy as pb

# local imports
from score import read_score
from state import parse_empty_blocks
from state import build_block_map
from state import find_empty_blocks

from piece import erase_piece

class GBGym(Env):
    def __init__(self, *, device='cpu', speed=0, live_feed=False):
        self.live_feed = live_feed
        self.ticks = 0
        self.device = device
        self.game_over_screen = np.load("game_over.npy")

        self.gameboy = start_gameboy(speed)
        self.sm = self.gameboy.botsupport_manager()

        # different actions possible...
        self.action_space = np.arange(0, 7)
        self.current_score = 0

        if live_feed:
            import matplotlib.pyplot as plt
            ax = plt.subplot(1, 1, 1)
            self.image_feed = ax
            plt.ion()
            plt.show()

    def score(self):
        game_score = read_score(self.sm.screen().screen_ndarray())
        return game_score

    # step moves forward two frames...
    def step(self, action):
        self.ticks += 1
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

        # get numpy array that represents pixels...
        # chop out all the details other than the board...
        block_map = build_block_map(self.sm.screen().screen_ndarray())

        piece_state = _get_piece_state(self.gameboy)
        piece_vector = piece_state.to_vector()

        # If emtpy block score is zero just use 1
        empty_block_score = torch.sum(find_empty_blocks(block_map, piece_state)) or 1

        # reward is how much the score improved...
        reward = new_score / empty_block_score
        self.current_score = new_score

        if self.live_feed:
            if self.ticks % 2 == 0:
                erased_floating_piece = erase_piece(
                        block_map=np.copy(block_map),
                        rotation=piece_state.rotation_position,
                        piece=piece_state.current_piece,
                        position=(piece_state.x, piece_state.y)
                    )
                self.image_feed.imshow(erased_floating_piece)

        observation = np.concatenate([[piece_vector], block_map])
        observation = torch.tensor([observation], device=self.device, dtype=torch.float32)

        truncated = False
        terminated = self.is_game_over()

        if terminated:
            action_reward = 0

        # added extra
        return (observation, reward, terminated, truncated)

    # kinda janky but it hasn't failed me yet...
    def is_game_over(self):
        screen = self.sm.screen().screen_ndarray()
        seg1 = screen[0, 16]
        seg2 = screen[0, 32]
        seg3 = screen[0, 64]
        red_value = np.array([0, 0, 248])

        return (
            np.all(seg1 == red_value)
            or np.all(seg2 == red_value)
            or np.all(seg3 == red_value)
        )

    def close(self):
        self.gameboy.stop()

        if self.live_feed:
            plt.ioff()

    def reset(self):
        # just return an empty dict because info isn't being used...
        info = dict()

        f = open("start2.state", "rb")
        self.gameboy.load_state(f)
        f.close()
        block_map = build_block_map(self.sm.screen().screen_ndarray())

        piece_state = _get_piece_state(self.gameboy).to_vector()
        state = np.concatenate([[piece_state], block_map])
        state = torch.tensor([state], device=self.device, dtype=torch.float32)

        return (state, info)

@dataclass
class PieceState:
    x: int
    y: int
    next_piece: int
    current_piece: int
    previous_piece: int
    rotation_position: int

    def to_vector(self):
        return np.array(
            [
                self.x,
                self.y,
                self.next_piece,
                self.previous_piece,
                self.current_piece,
                self.rotation_position,
                0,
                0,
                0,
                0,
            ]
        )

# returns vector composed of numbers representing
# [x_position, y_position, rotation_state, previous_piece, current_piece, next_piece]
def _get_piece_state(gb):
    PIECE_POSITION_X_ADDR = 0xAF80
    PIECE_POSITION_Y_ADDR = 0xAF81

    PIECE_ROTATION_POSITION_ADDR = 0xAF82
    NEXT_PIECE_ADDR = 0xAF84
    CURRENT_PIECE_ADDR = 0xAF85
    PREVIOUS_PIECE_ADDR = 0xAFB0

    x = gb.get_memory_value(PIECE_POSITION_X_ADDR)
    y = gb.get_memory_value(PIECE_POSITION_Y_ADDR)

    rotation_position = gb.get_memory_value(PIECE_ROTATION_POSITION_ADDR)
    next_piece = gb.get_memory_value(NEXT_PIECE_ADDR)
    current_piece = gb.get_memory_value(CURRENT_PIECE_ADDR)
    previous_piece = gb.get_memory_value(PREVIOUS_PIECE_ADDR)

    return PieceState(
        x, y, next_piece, current_piece, previous_piece, rotation_position
    )

def main():
    with start_gameboy(1) as gb:
        sm = gb.botsupport_manager()
        game_over_screen = np.load("game_over.npy")

        piece_state = _get_piece_state(gb)
        old_game_score = 0

        wait_n_seconds(gb, 2)

        while not gb.tick():
            time.sleep(0.05)
            screen = sm.screen().screen_ndarray()

            seg1 = screen[0, 16]
            seg2 = screen[0, 32]
            seg3 = screen[0, 64]
            seg4 = screen[0, 64]
            red_value = np.array([0, 0, 248])

            is_game_over = (
                np.all(seg3 == red_value)
                or np.all(seg2 == red_value)
                or np.all(seg4 == red_value)
                or np.all(seg1 == red_value)
            )

            new_piece_state = _get_piece_state(gb)
            if new_piece_state != piece_state:
                piece_state = new_piece_state
                print(piece_state)

            if is_game_over:
                print("GAME IS OVER :(")
                break

        fig = px.imshow(build_block_map(screen))
        fig.show()


def start_gameboy(speed=1):
    gb = pb.PyBoy("tetris_dx.sgb")

    gb.set_emulation_speed(target_speed=speed)

    f = open("start2.state", "rb")
    gb.load_state(f)
    f.close()

    gb.send_input(pb.WindowEvent.PRESS_BUTTON_START)
    gb.tick()
    gb.send_input(pb.WindowEvent.RELEASE_BUTTON_START)
    gb.tick()

    return gb

def seconds_to_frames(seconds):
    return seconds * FPS


def wait_n_seconds(gb, n):
    i = 0
    while not gb.tick() and i < int(seconds_to_frames(n)):
        i += 1


if __name__ == "__main__":
    main()
