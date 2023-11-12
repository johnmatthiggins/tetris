import numpy as np


def erase_piece(block_map, rotation, piece, position):
    x, y = position

    # decrement X because it's not zero indexed.
    x -= 1

    match piece:
        # I block
        case 0:
            _erase_I_block(block_map, x, y, rotation)
        # T block
        case 1:
            _erase_T_block(block_map, x, y, rotation)
        # Z block
        case 2:
            _erase_Z_block(block_map, x, y, rotation)
        # S block
        case 3:
            _erase_S_block(block_map, x, y, rotation)
        # J block
        case 4:
            _erase_J_block(block_map, x, y, rotation)
        # L block
        case 5:
            _erase_L_block(block_map, x, y, rotation)
        # Square block
        case 6:
            _erase_square_block(block_map, x, y)

    return block_map


def _erase_square_block(block_map, x, y):
    block_map[y, x + 1] = 0
    block_map[y, x] = 0
    block_map[y + 1, x + 1] = 0
    block_map[y + 1] = 0


def _erase_I_block(block_map, x, y, rotation):
    # already touching the bottom so it doesn't need to be removed...
    if y == 18:
        return

    match rotation:
        case 0 | 2:
            if block_map[y, x] == 1:
                block_map[y, x] = 0
                block_map[y, x - 1] = 0
                block_map[y, x + 1] = 0
                block_map[y, x + 2] = 0
            else:
                # check if it's going to be touching the bottom,
                # if it is, ignore it because it won't effect reward calculation.
                if y < 17:
                    y += 1
                    block_map[y, x] = 0
                    block_map[y, x - 1] = 0
                    block_map[y, x + 1] = 0
                    block_map[y, x + 2] = 0
        case 1 | 3:
            if rotation == 3:
                x += 1

            if y != 0:
                block_map[y - 1, x] = 0
            block_map[y, x] = 0
            block_map[y + 1, x] = 0
            block_map[y + 2, x] = 0


def _erase_T_block(block_map, x, y, rotation):
    match rotation:
        case 0:
            block_map[y, x] = 0
            block_map[y, x + 1] = 0
            block_map[y + 1, x] = 0
            block_map[y, x - 1] = 0
        case 1:
            block_map[y, x] = 0
            block_map[y - 1, x] = 0
            block_map[y + 1, x] = 0
            block_map[y, x - 1] = 0
        case 2:
            block_map[y, x] = 0
            block_map[y, x + 1] = 0
            block_map[y - 1, x] = 0
            block_map[y, x - 1] = 0
        case 3:
            block_map[y, x] = 0
            block_map[y - 1, x] = 0
            block_map[y + 1, x] = 0
            block_map[y, x + 1] = 0


def _erase_Z_block(block_map, x, y, rotation):
    # match rotation:
    #     case 0:
    #         block_map[y, x]         = 0
    #         block_map[y, x - 1]     = 0
    #         block_map[y + 1, x + 1] = 0
    #         block_map[y + 1, x]     = 0
    #     case 1:
    #         block_map[y, x]         = 0
    #         block_map[y, x - 1]     = 0
    #         block_map[y - 1, x - 1] = 0
    #         block_map[y + 1, x]     = 0
    #     case 2:
    #         block_map[y - 1, x]     = 0
    #         block_map[y - 1, x - 1] = 0
    #         block_map[y, x + 1]     = 0
    #         block_map[y, x]         = 0
    #     case 3:
    #         block_map[y - 1, x]     = 0
    #         block_map[y, x]         = 0
    #         block_map[y, x + 1]     = 0
    #         block_map[y + 1, x + 1] = 0
    for i in [1, 0, -1]:
        for j in [1, 0, -1]:
            block_map[y + j, x + i] = 0


def _erase_S_block(block_map, x, y, rotation):
    match rotation:
        case 0:
            block_map[y, x] = 0
            block_map[y, x + 1] = 0
            block_map[y - 1, x] = 0
            block_map[y - 1, x - 1] = 0
        case 1:
            block_map[y, x] = 0
            block_map[y + 1, x] = 0
            block_map[y, x - 1] = 0
            block_map[y - 1, x - 1] = 0
        case 2:
            y += 1
            block_map[y, x] = 0
            block_map[y, x + 1] = 0
            block_map[y - 1, x] = 0
            block_map[y - 1, x - 1] = 0
        case 3:
            block_map[y - 1, x] = 0
            block_map[y, x] = 0
            block_map[y, x + 1] = 0
            block_map[y + 1, x + 1] = 0


def _erase_J_block(block_map, x, y, rotation):
    match rotation:
        case 0:
            block_map[y - 1, x] = 0
            block_map[y - 1, x - 1] = 0
            block_map[y - 1, x + 1] = 0
            block_map[y, x + 1] = 0
        case 1:
            block_map[y + 1, x] = 0
            block_map[y, x] = 0
            block_map[y - 1, x] = 0
            block_map[y + 1, x - 1] = 0
        case 2:
            block_map[y - 1, x - 1] = 0
            block_map[y, x - 1] = 0
            block_map[y, x] = 0
            block_map[y, x + 1] = 0
        case 3:
            block_map[y - 1, x + 1] = 0
            block_map[y - 1, x] = 0
            block_map[y, x] = 0
            block_map[y + 1, x] = 0


def _erase_L_block(block_map, x, y, rotation):
    match rotation:
        case 0:
            block_map[y, x] = 0
            block_map[y, x - 1] = 0
            block_map[y, x + 1] = 0
            block_map[y + 1, x - 1] = 0
        case 1:
            block_map[y + 1, x] = 0
            block_map[y, x] = 0
            block_map[y - 1, x] = 0
            block_map[y - 1, x - 1] = 0
        case 2:
            block_map[y, x - 1] = 0
            block_map[y, x] = 0
            block_map[y, x + 1] = 0
            block_map[y - 1, x + 1] = 0
        case 3:
            block_map[y - 1, x] = 0
            block_map[y, x] = 0
            block_map[y + 1, x] = 0
            block_map[y + 1, x + 1] = 0
