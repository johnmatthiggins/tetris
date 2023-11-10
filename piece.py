import numpy as np

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
