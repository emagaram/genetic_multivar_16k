import math
import sys
import time
from redirects import RedirectsEvaluator
from settings import ADJACENT_BONUS, BAD_REDIRECT, INWARD_BONUS
from keyboard import Key, Keyboard
from util import kb_to_column_dict, kb_to_reverse_column_dict, kb_to_row_dict
from words import CorpusFrequencies, get_trigrams


# Old class, not used

class RollEvaluator:
    def generate_roll_indexes(
        n,
    ) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
        result = []
        # 0, 1, 4 => [(0, 1, 4), (1, 0, 4), (4,0,1), (4,1,0)]
        # Generate every pair going both directions for both hands
        # Loop through 0-4
        # Loop through 0 - 4
        #   Generate every pair going both directions for both hands
        # for order in range(0,2):

        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                for k in range(0, 4):
                    result.append(((0, i), (0, j), (1, k)))
                    result.append(((0, j), (0, i), (1, k)))
                    result.append(((1, i), (1, j), (0, k)))
                    result.append(((1, j), (1, i), (0, k)))
        for k in range(0, 4):
            for i in range(n + 1):
                for j in range(i + 1, n + 1):
                    result.append(((0, k), (1, i), (1, j)))
                    result.append(((0, k), (1, j), (1, i)))
                    result.append(((1, k), (0, j), (0, i)))
                    result.append(((1, k), (0, i), (0, j)))
        # for i,roll in enumerate(result):
        #     print(roll)
        #     if i % 4 == 3:
        #         print()
        return result

    POSITIONS_TO_EVAL: list[
        tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    ] = generate_roll_indexes(3)

    def __init__(self, kb: Keyboard = None) -> None:
        if kb:
            self.set_kb(kb)
        self.trigrams = get_trigrams()
        self.corpus_frequencies = CorpusFrequencies()

    def set_kb(self, kb: Keyboard):
        self.column_dict = kb_to_column_dict(kb)
        self.row_dict = kb_to_row_dict(kb)
        self.reverse_column_dict = kb_to_reverse_column_dict(kb)

    def evaluate_fast(self, max: float = sys.float_info.max) -> float:
        return self.evaluate_fast_inner(False, max)

    def evaluate_fast_stat(self, max: float = sys.float_info.max) -> float:
        return self.evaluate_fast_inner(True, max)

    def is_same_row_or_longer_higher(
        self, hand: int, col1: int, col2: int, row1: int, row2: int
    ):
        height = [[0, 1, 2, 1], [1, 2, 1, 0]]
        c1_height = height[hand][col1]
        c2_height = height[hand][col2]
        row_diff = row2 - row1
        height_diff = c2_height - c1_height
        return row1 == row2 or (-row_diff > 0) == (height_diff > 0)

    def get_hand_pos(self, trigram: str) -> tuple[int, int, int]:
        res = []
        for char in trigram:
            res.append(1 if self.column_dict[char] > 3 else 0)
        return tuple(res)

    def on_same_hand(self, col0: int, col1: int):
        return (col0 <= 3 and col1 <= 3) or (col0 > 3 and col1 > 3)

    def evaluate_slow(self):
        score = 0
        seen = set()
        for trigram, freq in self.trigrams.items():
            c1, c2, c3 = (
                self.column_dict[trigram[0]],
                self.column_dict[trigram[1]],
                self.column_dict[trigram[2]],
            )
            if c1 != c2 != c3 and (
                (self.on_same_hand(c1, c2) and not self.on_same_hand(c2, c3))
                or (self.on_same_hand(c2, c3) and not self.on_same_hand(c1, c3))
            ):
                score += freq
                seen.add(trigram)
        return seen, score / self.corpus_frequencies.trigrams_freq

    def evaluate_fast_inner(self, is_stat: bool, max=sys.float_info.max) -> float:
        hand_len = 4
        hand_column_groups = [
            [self.reverse_column_dict[i] for i in range(hand_len)],
            [self.reverse_column_dict[i] for i in range(hand_len, 2 * hand_len)],
        ]

        def is_pinky(hand: int, col: int):
            return (hand == 0 and col == 0) or (hand == 1 and col == 3)

        def is_ring(hand: int, col: int):
            return (hand == 0 and col == 1) or (hand == 1 and col == 2)

        def is_adjacent(col1: int, col2: int):
            return abs(col2 - col1) == 1

        def is_inward(hand: int, i1: int, i2: int):
            return (hand == 1 and i1 > i2) or (hand == 0 and i2 > i1)

        res = 0
        for (
            (i_hand, i_col),
            (j_hand, j_col),
            (k_hand, k_col),
        ) in RollEvaluator.POSITIONS_TO_EVAL:

            for char1 in hand_column_groups[i_hand][i_col]:
                for char2 in hand_column_groups[j_hand][j_col]:
                    for char3 in hand_column_groups[k_hand][k_col]:
                        trigram = char1 + char2 + char3
                        rolling_hand = j_hand
                        i1, i2 = (
                            (self.column_dict[char1], self.column_dict[char2])
                            if self.on_same_hand(
                                self.column_dict[char1], self.column_dict[char2]
                            )
                            else (self.column_dict[char2], self.column_dict[char3])
                        )

                        c1, c2 = (
                            (char1, char2)
                            if self.on_same_hand(
                                self.column_dict[char1], self.column_dict[char2]
                            )
                            else (char2, char3)
                        )
                        row1 = self.row_dict[c1]
                        row2 = self.row_dict[c2]
                        if not is_stat and is_pinky(rolling_hand, i1 % 4):
                            continue
                        if not is_stat and (
                            is_ring(rolling_hand, i1 % 4)
                            and is_pinky(rolling_hand, i2 % 4)
                        ):
                            continue
                        if not is_stat and not self.is_same_row_or_longer_higher(
                            rolling_hand, i1 % 4, i2 % 4, row1, row2
                        ):
                            continue
                        add = self.trigrams.get(trigram)
                        if add != None:
                            if not is_stat:
                                if is_inward(rolling_hand, i1 % 4, i2 % 4):
                                    add *= INWARD_BONUS
                                if is_adjacent(i1, i2):
                                    add *= ADJACENT_BONUS
                            res += add
        return res / self.corpus_frequencies.trigrams_freq


def test_roll():
    keyboard: Keyboard = Keyboard(
        [[2, 2, 2, 2], [2, 2, 2, 2]],
        kb=[
            [
                [Key("."), Key("hu")],
                [Key(","), Key("ef")],
                [Key("'ox"), Key("az")],
                [Key("ik"), Key("py")],
            ],
            [
                [Key("n"), Key("r")],
                [Key("cl"), Key("gm")],
                [Key("dw"), Key("qt")],
                [Key("bv"), Key("js")],
            ],
        ],
    )
    rolls_evaluator = RollEvaluator(keyboard)
    rolls = rolls_evaluator.evaluate_fast()
    fs = time.time()
    rolls_fast_stat = rolls_evaluator.evaluate_fast_stat()
    fe = time.time()
    ss = time.time()
    _, rolls_slow = rolls_evaluator.evaluate_slow()
    se = time.time()
    # assert rolls_evaluator.is_same_row_or_longer_higher(0, 0, 1, 0, 0) == True
    # assert rolls_evaluator.is_same_row_or_longer_higher(0, 0, 1, 0, 1) == False
    redirect_evaluator = RedirectsEvaluator(keyboard)
    redirect_stat = 0
    for trigram in redirect_evaluator.trigrams.items():
        redirect_stat += redirect_evaluator.evaluate_trigram_stat(trigram)
    # print(f"Redirect stats: {redirect_stat}")
    # print("Redirect tests passed!")


test_roll()
