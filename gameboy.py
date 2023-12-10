#!/usr/bin/env python3
from dataclasses import dataclass
import time
import sys
import io

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
from score import read_lines
from state import build_block_map
from state import find_empty_blocks
from state import bumpiness_score

FPS = 60
GAME_OVER = np.load("game_over.npy")


class GBGym(Env):
    def __init__(self, *, device="cpu", speed=0, live_feed=False, step_backwards=False):
        # add functionality for stepping backwards...
        self.step_backwards = step_backwards

        self.live_feed = live_feed
        self.device = device
        self.gameboy = start_gameboy(speed)
        self.sm = self.gameboy.botsupport_manager()

        # different actions possible...
        self.action_space = np.arange(0, 44)

    def score(self):
        game_score = read_score(self.sm.screen().screen_ndarray())
        return game_score

    def lines(self):
        game_score = read_lines(self.sm.screen().screen_ndarray())
        return game_score

    # step moves forward two frames...
    def step(self, action):
        # save state if stepping backwards is enabled...
        if self.step_backwards:
            with open('back.state', 'wb') as f:
                self.gameboy.save_state(f)

        _make_move(action, self.gameboy, self.sm)

        point_score = self.score()
        line_score = self.lines()

        # get numpy array that represents pixels...
        # chop out all the details other than the board...
        block_map = build_block_map(self.sm.screen().screen_ndarray())

        piece_state = _get_piece_state(self.gameboy)
        piece_vector = piece_state.to_vector()

        bumpiness, bump_vector = bumpiness_score(block_map)
        height = np.sum(bump_vector)

        empty_blocks = find_empty_blocks(block_map).sum()

        # new_aggregated_score = point_score + 10 * (line_score) - (bumpiness * 1) - (empty_blocks * 10)
        new_aggregated_score = (point_score - 0.51 * height + 0.76 * line_score
                                - 10 * empty_blocks - 0.18 * bumpiness)
        reward = new_aggregated_score - self.current_aggregated_score

        self.prev_aggregated_score = self.current_aggregated_score
        self.current_aggregated_score = new_aggregated_score

        observation = np.concatenate([piece_vector, bump_vector])
        observation = torch.tensor(
            [observation], device=self.device, dtype=torch.float32
        )

        truncated = False
        terminated = self.is_game_over()

        if terminated:
            action_reward = 0

        # added extra
        return (observation, reward, terminated, truncated)
    
    def step_back(self):
        self.current_aggregated_score = self.prev_aggregated_score
        self.previous_aggregated_score = 0

        with open('back.state', 'rb') as f:
            self.gameboy.load_state(f)


    # kinda janky but it hasn't failed me yet...
    def is_game_over(self):
        screen = self.sm.screen().screen_ndarray()
        return _is_game_over(screen)

    def close(self):
        self.gameboy.stop()

        if self.live_feed:
            plt.ioff()

    def reset(self):
        self.current_aggregated_score = 0
        self.prev_aggregated_score = 0

        # just return an empty dict because info isn't being used...
        info = dict()

        n = np.random.randint(low=0, high=3)
        f = open(f"start{n}.state", "rb")

        self.gameboy.load_state(f)
        f.close()
        block_map = build_block_map(self.sm.screen().screen_ndarray())

        _, bump_vector = bumpiness_score(block_map)

        piece_state = _get_piece_state(self.gameboy).to_vector()

        observation = np.concatenate([piece_state, bump_vector])
        observation = torch.tensor(
            [observation], device=self.device, dtype=torch.float32
        )
        print(observation.shape)

        return (observation, info)


def _is_game_over(rgb_screen):
    seg1 = rgb_screen[0, 16]
    seg2 = rgb_screen[0, 24]
    seg3 = rgb_screen[0, 32]
    seg4 = rgb_screen[0, 40]
    seg4 = rgb_screen[0, 48]
    seg5 = rgb_screen[0, 56]
    seg6 = rgb_screen[0, 64]
    red_value = np.array([0, 0, 248])

    return (
        np.all(seg1 == red_value)
        or np.all(seg2 == red_value)
        or np.all(seg3 == red_value)
        or np.all(seg4 == red_value)
        or np.all(seg5 == red_value)
        or np.all(seg6 == red_value)
        or _is_game_over_screen_showing(rgb_screen)
    )


def _is_game_over_screen_showing(rgb_screen):
    relevant_part = rgb_screen[21 : 68 + 1, 29 : 84 + 1]
    return np.all(relevant_part == GAME_OVER)


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
        x=x,
        y=y,
        next_piece=next_piece,
        current_piece=current_piece,
        previous_piece=previous_piece,
        rotation_position=rotation_position,
    )


def _make_move(move, gb, sm):
    piece_state = _get_piece_state(gb)
    position, rotations = _decode_move(move)

    for _ in range(rotations):
        gb.send_input(pb.WindowEvent.PRESS_BUTTON_A)
        gb.tick()
        gb.tick()
        gb.send_input(pb.WindowEvent.RELEASE_BUTTON_A)
        gb.tick()

    if position != 0:
        if position > 5:
            for _ in range(position - 5):
                gb.send_input(pb.WindowEvent.PRESS_ARROW_LEFT)
                gb.tick()
                gb.tick()
                gb.send_input(pb.WindowEvent.RELEASE_ARROW_LEFT)
                gb.tick()
        else:
            for _ in range(position):
                gb.send_input(pb.WindowEvent.PRESS_ARROW_RIGHT)
                gb.tick()
                gb.tick()
                gb.send_input(pb.WindowEvent.RELEASE_ARROW_RIGHT)
                gb.tick()

    new_piece_state = _get_piece_state(gb)
    screen = sm.screen().screen_ndarray()
    while (
        new_piece_state.current_piece == piece_state.current_piece
        and new_piece_state.next_piece == piece_state.next_piece
        and new_piece_state.previous_piece == piece_state.previous_piece
        and not _is_game_over(screen)
    ):
        gb.send_input(pb.WindowEvent.PRESS_ARROW_DOWN)
        gb.tick()

        new_piece_state = _get_piece_state(gb)
        screen = sm.screen().screen_ndarray()

    gb.send_input(pb.WindowEvent.RELEASE_ARROW_DOWN)
    gb.tick()


def _decode_move(move):
    horizontal_position = (move & 0x3C) >> 2
    rotations = move & 0x03

    return (horizontal_position, rotations)


def main():
    if "--live-feed" in sys.argv:
        import matplotlib.pyplot as plt

        ax = plt.subplot(1, 1, 1)
        image_feed = ax
        plt.ion()
        plt.show()

    with start_gameboy(1) as gb:
        sm = gb.botsupport_manager()
        game_over_screen = np.load("game_over.npy")

        piece_state = _get_piece_state(gb)
        old_game_score = 0

        # wait_n_seconds(gb, 2)

        while not gb.tick():
            screen = sm.screen().screen_ndarray()

            block_map = build_block_map(screen)
            piece_state = _get_piece_state(gb)

            if "--live-feed" in sys.argv:
                erased_floating_piece = erase_piece(
                    block_map=np.copy(block_map),
                    rotation=piece_state.rotation_position,
                    piece=piece_state.current_piece,
                    position=(piece_state.x, piece_state.y),
                )
                image_feed.imshow(erased_floating_piece)

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


def start_gameboy(speed=1):
    gb = pb.PyBoy("tetris_dx.sgb")

    gb.set_emulation_speed(target_speed=speed)

    n = np.random.randint(low=0, high=3)
    f = open(f"start{n}.state", "rb")
    gb.load_state(f)
    f.close()

    return gb


def seconds_to_frames(seconds):
    return seconds * FPS


def wait_n_seconds(gb, n):
    i = 0
    while not gb.tick() and i < int(seconds_to_frames(n)):
        i += 1


if __name__ == "__main__":
    main()
