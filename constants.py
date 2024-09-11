from enum import Enum
import sys

class Categories(Enum):
    INACCURACY = "inaccuracy"
    REDIRECT = "redirect"
    SFB = "sfb"
    SFS = "sfs"
    DISCOMFORT = "discomfort"
    FINGERFREQ = "fingerfreq"

INACCURACY_WEIGHT = 0.3
SFB_WEIGHT = 0.15
REDIRECT_WEIGHT = 0.2
FINGER_FREQ_WEIGHT = 0.2
DISCOMFORT_WEIGHT = 0.1
SFS_WEIGHT = 0.05

SFB_SFS_DIFF_KEY_PENALTY = 3
SFB_SFS_PINKY_PUNISHMENT = 3
GOAL_FINGER_FREQ = [
    [0.18 * 0.815, 0.155 * 0.815, 0.115 * 0.815, 0.05 * 0.815],
    [0.18 * 1.185, 0.155 * 1.185, 0.115 * 1.185, 0.05 * 1.185],
]  # Left should have 40.75% to account for space key addition


assert (
    abs(
        INACCURACY_WEIGHT
        + SFB_WEIGHT
        + REDIRECT_WEIGHT
        + FINGER_FREQ_WEIGHT
        + DISCOMFORT_WEIGHT
        + SFS_WEIGHT
        - 1
    )
    < sys.float_info.epsilon
)
