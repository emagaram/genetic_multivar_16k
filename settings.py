from enum import Enum

class InaccuracyMode(Enum):
    IGNORE_FIRST = "Ignore First"
    IGNORE_LAST = "Ignore Last"
    ONLY_FIRST = "Only First"
    ONLY_LAST = "Only Last"
    MIDDLE = "Middle"
    ALL = "All"


PRINT = True
MODE = InaccuracyMode.IGNORE_FIRST
USE_PUNCTUATION = True
NUM_MAGIC = 0
INACCURACY_WEIGHT = 1
SFB_WEIGHT = 0.1
DISCOMFORT_WEIGHT = 0.025
SFS_WEIGHT = 0.025
REDIRECT_WEIGHT = 0.025
FINGER_FREQ_WEIGHT = 0.0075 # 1% SFB = 13% adjustment, seems like a lot but if the pinky is at 8%, 8*1.13 = 9.04%. Fair "price" to pay for 1% SFB

# Discomfort
PINKY_ABOVE_INDEX = 1.1
PINKY_ABOVE_MIDDLE = 1.1**2
PINKY_ABOVE_RING = 1.1**3
RING_ABOVE_MIDDLE = 1.1**3
OUTWARD = 1.05

# Redirect
BAD_REDIRECT = 2

# SFB SFS
# Need this penalty since SFRs count as an SFB (a "0U SFB"). SFRs count as 1/SFB_SFS_DIFF_KEY_PENALTY of an SFB
SFB_SFS_DIFF_KEY_PENALTY = 3
SFB_SFS_PINKY_PENALTY = 2.5

# Rolls
INWARD_BONUS = 1.1
ADJACENT_BONUS = 1.05

# Finger Freq
PINKY_WEIGHT = 1.5
RING_WEIGHT = 3.6
MIDDLE_WEIGHT = 4.8
INDEX_WEIGHT = 5.5

# From Oxeylyzer, pinky - 1%
PINKY_MAX = 0.08
RING_MAX = 0.14
MIDDLE_MAX = 0.2
INDEX_MAX = 0.2
GOAL_FINGER_MAX = [[PINKY_MAX, RING_MAX, MIDDLE_MAX, INDEX_MAX],[INDEX_MAX, MIDDLE_MAX, RING_MAX, PINKY_MAX]]
weight_sum = PINKY_WEIGHT + RING_WEIGHT + MIDDLE_WEIGHT + INDEX_WEIGHT
PINKY_FREQ = PINKY_WEIGHT / (2 * weight_sum)
RING_FREQ = RING_WEIGHT / (2 * weight_sum)
MIDDLE_FREQ = MIDDLE_WEIGHT / (2 * weight_sum)
INDEX_FREQ = INDEX_WEIGHT / (2 * weight_sum)
GOAL_FINGER_FREQ = [
    [PINKY_FREQ, RING_FREQ, MIDDLE_FREQ, INDEX_FREQ],
    [INDEX_FREQ, MIDDLE_FREQ, RING_FREQ, PINKY_FREQ],
]
assert abs(sum(key for hand in GOAL_FINGER_FREQ for key in hand)) - 1 < 0.00001


def settings_to_str(space: str = " ") -> str:
    return f"""Settings:
INACCURACY_WEIGHT = {INACCURACY_WEIGHT}
SFB_WEIGHT = {SFB_WEIGHT}
REDIRECT_WEIGHT = {REDIRECT_WEIGHT}
DISCOMFORT_WEIGHT = {DISCOMFORT_WEIGHT}
SFS_WEIGHT = {SFS_WEIGHT}
FINGER_FREQ_WEIGHT = {FINGER_FREQ_WEIGHT}
FINGER_FREQ_WEIGHT = {FINGER_FREQ_WEIGHT}

Discomfort:
{space}PINKY_ABOVE_RING = {PINKY_ABOVE_RING}
{space}PINKY_ABOVE_MIDDLE = {PINKY_ABOVE_MIDDLE}
{space}RING_ABOVE_MIDDLE = {RING_ABOVE_MIDDLE}

Redirect:
{space}BAD_REDIRECT = {BAD_REDIRECT}

SFB SFS:
{space}SFB_SFS_DIFF_KEY_PENALTY = {SFB_SFS_DIFF_KEY_PENALTY}
{space}SFB_SFS_PINKY_PENALTY = {SFB_SFS_PINKY_PENALTY}

Good Rolls:
{space}INWARD_BONUS = 1.1
{space}ADJACENT_BONUS = 1.05

Finger Frequencies:
{space}PINKY_MAX = {PINKY_MAX}
{space}RING_MAX = {RING_MAX}
{space}MIDDLE_MAX = {MIDDLE_MAX}
{space}INDEX_MAX = {INDEX_MAX}

Rolls
INWARD_BONUS = {INWARD_BONUS}
ADJACENT_BONUS = {ADJACENT_BONUS}
"""
