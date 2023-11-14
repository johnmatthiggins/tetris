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
from score import read_lines
from state import build_block_map
from state import find_empty_blocks
from state import bumpiness_score

FPS = 60


class GBGym(Env):
    def __init__(self, *, device="cpu", speed=0, live_feed=False):
        self.live_feed = live_feed
        self.device = device
        self.game_over_screen = np.load("game_over.npy")

        self.gameboy = start_gameboy(speed)
        self.sm = self.gameboy.botsupport_manager()

        # different actions possible...
        self.action_space = np.arange(0, 44)
        self.current_score = 0
        self.current_lines = 0
        self.current_bumpiness = 0
        self.current_emptiness = 0

        if live_feed:
            import matplotlib.pyplot as plt

            ax = plt.subplot(1, 1, 1)
            self.image_feed = ax
            plt.ion()
            plt.show()

    def score(self):
        game_score = read_score(self.sm.screen().screen_ndarray())
        return game_score

    def lines(self):
        game_score = read_lines(self.sm.screen().screen_ndarray())
        return game_score

    # step moves forward two frames...
    def step(self, action):
        _make_move(action, self.gameboy, self.sm)

        new_score = self.score()
        new_line_count = self.lines()

        line_diff = new_line_count - self.current_lines
        score_diff = new_score - self.current_score

        # get numpy array that represents pixels...
        # chop out all the details other than the board...
        block_map = build_block_map(self.sm.screen().screen_ndarray())

        piece_state = _get_piece_state(self.gameboy)
        piece_vector = piece_state.to_vector()

        bumpiness = bumpiness_score(block_map)
        bump_diff = bumpiness - self.current_bumpiness

        self.current_bumpiness = bumpiness

        empty_blocks = find_empty_blocks(block_map)
        empty_diff = empty_blocks.sum() - self.current_emptiness

        self.current_emptiness = empty_blocks.sum()

        reward = score_diff + 10 * (line_diff) - bump_diff - empty_diff

        print('*' * 10)
        print('EMPTY_BLOCKS: %s' % str(empty_blocks))
        print('REWARD: %s' % str(reward))
        print('*' * 10)

        self.current_score = new_score
        self.current_lines = new_line_count

        if self.live_feed:
            self.image_feed.imshow(block_map_minus_top_two_lines)

        observation = np.concatenate([[piece_vector], block_map])
        observation = torch.tensor(
            [observation], device=self.device, dtype=torch.float32
        )

        truncated = False
        terminated = self.is_game_over()

        if terminated:
            action_reward = 0

        # added extra
        return (observation, reward, terminated, truncated)

    # kinda janky but it hasn't failed me yet...
    def is_game_over(self):
        screen = self.sm.screen().screen_ndarray()
        return _is_game_over(screen)

    def close(self):
        self.gameboy.stop()

        if self.live_feed:
            plt.ioff()

    def reset(self):
        self.current_score = 0
        self.current_lines = 0
        self.current_bumpiness = 0
        self.current_emptiness = 0

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
    )


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

        wait_n_seconds(gb, 2)

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

        print(read_lines(screen))
        print(read_score(screen))
        fig = px.imshow(screen)
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
