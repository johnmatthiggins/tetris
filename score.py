import cv2
import numpy as np
import plotly.express as px

SCORE_LEFT_X = 112
SCORE_RIGHT_X = 159
SCORE_TOP_Y = 80
SCORE_BTM_Y = 86

LINES_LEFT_X = 112
LINES_RIGHT_X = 150
LINES_TOP_Y = 128
LINES_BTM_Y = 134

# BLACK PIXEL VALUE
# I shortened it because i was tired of typing it.
BPV = 248

ZEROS_ARRAY = np.zeros(shape=(6, 7))

# 7x6 array representing the number zero on screen
_SCORE_ZERO = np.array(
    [
        [0, BPV, BPV, BPV, BPV, BPV, 0],
        [0, BPV, 0, 0, BPV, BPV, 0],
        [BPV, BPV, 0, 0, 0, BPV, BPV],
        [BPV, BPV, 0, 0, 0, BPV, BPV],
        [0, BPV, 0, 0, BPV, BPV, 0][::-1],
        [0, 0, BPV, BPV, BPV, 0, 0],
    ]
)

_SCORE_ONE = np.array(
    [
        [0, 0, 0, BPV, BPV, 0, 0],
        [0, 0, BPV, BPV, BPV, 0, 0],
        [0, 0, 0, BPV, BPV, 0, 0],
        [0, 0, 0, BPV, BPV, 0, 0],
        [0, 0, 0, BPV, BPV, 0, 0],
        [0, BPV, BPV, BPV, BPV, BPV, BPV],
    ]
)

_SCORE_TWO = np.array(
    [
        [0, BPV, BPV, BPV, BPV, BPV, 0],
        [BPV, BPV, 0, 0, 0, BPV, BPV],
        [0, 0, 0, 0, BPV, BPV, BPV],
        [0, BPV, BPV, BPV, BPV, 0, 0],
        [BPV, BPV, BPV, 0, 0, 0, 0],
        np.full(7, BPV),
    ]
)

_SCORE_THREE = np.array(
    [
        np.concatenate([[0], np.full(6, BPV)]),
        np.concatenate([np.full(4, 0), [BPV, BPV], [0]]),
        np.concatenate([[0, 0], np.full(3, BPV), [0, 0]]),
        np.concatenate([np.full(5, 0), [BPV, BPV]]),
        np.concatenate([[BPV, BPV], [0, 0, 0], [BPV, BPV]]),
        np.concatenate([[0], np.full(5, BPV), [0]]),
    ]
)

_SCORE_FOUR = np.array(
    [
        np.concatenate([np.zeros(3), np.full(3, BPV), [0]]),
        np.concatenate([np.zeros(2), np.full(4, BPV), [0]]),
        np.concatenate([[0], [BPV, BPV], [0], [BPV, BPV], [0]]),
        np.concatenate([[BPV, BPV], [0, 0], [BPV, BPV], [0]]),
        np.full(7, BPV),
        [0, 0, 0, 0, BPV, BPV, 0],
    ]
)

_SCORE_FIVE = np.array(
    [
        np.full(7, BPV),
        np.concatenate([[BPV, BPV], np.full(5, 0)]),
        np.concatenate([np.full(6, BPV), [0]]),
        np.concatenate([np.full(5, 0), [BPV, BPV]]),
        np.concatenate([[BPV, BPV], np.full(3, 0), [BPV, BPV]]),
        np.concatenate([[0], np.full(5, BPV), [0]]),
    ]
)

_SCORE_SIX = np.array(
    [
        np.concatenate([[0], np.full(5, BPV), [0]]),
        np.concatenate([[BPV, BPV], np.full(5, 0)]),
        np.concatenate([np.full(6, BPV), [0]]),
        np.concatenate([[BPV, BPV], np.zeros(3), [BPV, BPV]]),
        np.concatenate([[BPV, BPV], np.zeros(3), [BPV, BPV]]),
        np.concatenate([[0], np.full(5, BPV), [0]]),
    ]
)

_SCORE_SEVEN = np.array(
    [
        np.full(7, BPV),
        [BPV, BPV, 0, 0, 0, BPV, BPV],
        np.concatenate([np.full(4, 0), [BPV, BPV], [0]]),
        np.concatenate([np.full(3, 0), [BPV, BPV], [0, 0]]),
        np.concatenate([[0, 0], [BPV, BPV], [0, 0, 0]]),
        np.concatenate([[0, 0], [BPV, BPV], [0, 0, 0]]),
    ]
)

_SCORE_EIGHT = np.array(
    [
        np.concatenate([[0], np.full(5, BPV), [0]]),
        np.concatenate([[BPV, BPV], np.full(3, 0), [BPV, BPV]]),
        np.concatenate([[0], np.full(5, BPV), [0]]),
        np.concatenate([[BPV, BPV], np.full(3, 0), [BPV, BPV]]),
        np.concatenate([[BPV, BPV], np.full(3, 0), [BPV, BPV]]),
        np.concatenate([[0], np.full(5, BPV), [0]]),
    ]
)

_SCORE_NINE = np.array(
    [
        np.concatenate([[0], np.full(5, BPV), [0]]),
        np.concatenate([[BPV, BPV], np.full(3, 0), [BPV, BPV]]),
        np.concatenate([[BPV, BPV], np.full(3, 0), [BPV, BPV]]),
        np.concatenate([[0], np.full(6, BPV)]),
        np.concatenate([np.full(5, 0), np.full(2, BPV)]),
        np.concatenate([[0], np.full(5, BPV), [0]]),
    ]
)

_invert = np.vectorize(lambda n: 0 if n == BPV else BPV)


# pass in screen in color...
def read_score(screen):
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    score_part = screen[SCORE_TOP_Y : SCORE_BTM_Y + 1, SCORE_LEFT_X : SCORE_RIGHT_X + 2]

    return int(convert_score_to_string(score_part, 6))


# pass in screen in color...
def read_lines(screen):
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    score_part = screen[LINES_TOP_Y : LINES_BTM_Y + 1, LINES_LEFT_X : LINES_RIGHT_X + 2]
    return int(convert_score_to_string(score_part, 5))


def convert_score_to_string(score_segment, digit_count):
    NUMBER_WIDTH = 7 + 1
    NUMBER_HEIGHT = 6
    pixelated_numbers = ""

    for i in range(0, digit_count):
        x_offset = i * NUMBER_WIDTH

        pixelated_number = score_segment[:, x_offset + 1 : x_offset + NUMBER_WIDTH]
        truncated = pixelated_number[1:7, 0:7]

        pixelated_numbers += match_number_matrix(truncated)

    return pixelated_numbers


def convert_lines_to_string(score_segment):
    NUMBER_WIDTH = 7 + 1
    NUMBER_HEIGHT = 6
    pixelated_numbers = ""

    for i in range(0, 6):
        x_offset = i * NUMBER_WIDTH

        pixelated_number = score_segment[:, x_offset + 1 : x_offset + NUMBER_WIDTH]
        truncated = pixelated_number[1:7, 0:7]

        pixelated_numbers += match_number_matrix(truncated)

    return pixelated_numbers


def match_number_matrix(matrix):
    result = None

    if np.all(matrix - _invert(_SCORE_NINE) == 0):
        result = "9"
    elif np.all(matrix - _invert(_SCORE_EIGHT) == 0):
        result = "8"
    elif np.all(matrix - _invert(_SCORE_SEVEN) == 0):
        result = "7"
    elif np.all(matrix - _invert(_SCORE_SIX) == 0):
        result = "6"
    elif np.all(matrix - _invert(_SCORE_FIVE) == 0):
        result = "5"
    elif np.all(matrix - _invert(_SCORE_FOUR) == 0):
        result = "4"
    elif np.all(matrix - _invert(_SCORE_THREE) == 0):
        result = "3"
    elif np.all(matrix - _invert(_SCORE_TWO) == 0):
        result = "2"
    elif np.all(matrix - _invert(_SCORE_ONE) == 0):
        result = "1"
    else:
        result = "0"

    return result


if __name__ == "__main__":
    main()
