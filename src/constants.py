from enum import Enum


ROTATIONS_RANGE = (3, 5, 7)
SHIFTS_RANGE = (5, 7, 9)
ZOOM_RANGE = (1.05, 1.15, 1.25)
IMG_SLICE = (slice(0, 64), slice(0, 64))
DATA_DIRECTORY = '/tmp/facever'


class Mode(Enum):
    TRAIN = 1
    TEST = 2
