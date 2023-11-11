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
            pass
        # S block
        case 3:
            pass
        # J block
        case 4:
            _erase_J_block(block_map, x, y, rotation)
        # L block
        case 5:
            pass
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
    match rotation:
        case 0 | 2:
            if block_map[y, x] == 1:
                block_map[y, x]     = 0
                block_map[y, x - 1] = 0
                block_map[y, x + 1] = 0
                block_map[y, x + 2] = 0
            else:
                y += 1
                block_map[y, x]     = 0
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
            block_map[y, x]     = 0
            block_map[y, x + 1] = 0
            block_map[y + 1, x] = 0
            block_map[y, x - 1] = 0
        case 1:
            block_map[y, x]     = 0
            block_map[y - 1, x] = 0
            block_map[y + 1, x] = 0
            block_map[y, x - 1] = 0
        case 2:
            block_map[y, x]     = 0
            block_map[y, x + 1] = 0
            block_map[y - 1, x] = 0
            block_map[y, x - 1] = 0
        case 3:
            block_map[y, x]     = 0
            block_map[y - 1, x] = 0
            block_map[y + 1, x] = 0
            block_map[y, x + 1] = 0

def _erase_J_block(block_map, x, y, rotation):
    match rotation:
        case 0:
            block_map[y, x] = 0
            block_map[y, x - 1] = 0
            block_map[y, x + 1] = 0
            block_map[y + 1, x + 1] = 0
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

def _create_J_piece():
    j_matrix = np.zeros(shape=(5, 5))

    # make J shape
    j_matrix[1, 2] = 1
    j_matrix[2, 2] = 1
    j_matrix[3, 2] = 1
    j_matrix[3, 1] = 1

    return j_matrix


def _create_L_piece():
    l_matrix = np.flip(_create_J_piece(), axis=1)

    return l_matrix


def _create_square():
    square = np.zeros(shape=(5, 5))
    square[2, 1] = 1
    square[2, 3] = 1
    square[3, 3] = 1
    square[3, 2] = 1

    return square


def _create_I_piece():
    pass


def _create_T_piece():
    tpiece = np.zeros(shape=(5, 5))

    tpiece[2, 1] = 1
    tpiece[2, 2] = 1
    tpiece[2, 3] = 1
    tpiece[3, 2] = 1

    return tpiece


def _create_S_piece():
    pass


def _create_Z_piece():
    pass
