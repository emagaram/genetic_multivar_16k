import sys
import time
from settings import (
    OUTWARD,
    PINKY_ABOVE_INDEX,
    PINKY_ABOVE_MIDDLE,
    PINKY_ABOVE_RING,
    RING_ABOVE_MIDDLE,
)
from keyboard import Key, Keyboard
from util import kb_to_column_dict, kb_to_row_dict
from words import CorpusFrequencies, get_bigrams


class DiscomfortEvaluator:
    column_dict = {}
    row_dict = {}

    def __init__(self, kb: Keyboard = None) -> None:
        self.bigrams = get_bigrams()
        self.corpus_frequencies = CorpusFrequencies()
        if kb:
            self.set_kb(kb)

    def set_kb(self, kb: Keyboard):
        self.keyboard = kb
        self.column_dict = kb_to_column_dict(kb)
        self.row_dict = kb_to_row_dict(kb)

    def is_outward(self, hand: int, prev_col: int, current_col: int):
        return (hand == 0 and current_col < prev_col) or (
            hand == 1 and current_col > prev_col
        )

    def evaluate_bigram_inner_old(self, bigram: tuple[str, float], use_mult: bool):
        prev_char, curr_char = bigram[0][0], bigram[0][1]
        freq = bigram[1]
        prev_col = self.column_dict.get(prev_char)
        curr_col = self.column_dict.get(curr_char)
        # SFB_SFS evaluator will handle this
        if prev_col == curr_col:
            return 0
        curr_is_left = curr_col <= 3
        prev_is_left = prev_col <= 3

        prev_row = self.row_dict.get(prev_char)
        curr_row = self.row_dict.get(curr_char)
        # Keys are on different hand or we haven't changed rows
        if curr_is_left != prev_is_left or curr_row == prev_row:
            return 0
        curr_above_prev = curr_row < prev_row
        curr_below_prev = curr_row > prev_row
        curr_is_pinky = curr_col == 0 or curr_col == 7
        curr_is_ring = curr_col == 1 or curr_col == 6
        curr_is_index = curr_col == 3 or curr_col == 4
        curr_is_middle = curr_col == 2 or curr_col == 5
        prev_is_index = prev_col == 3 or prev_col == 4
        prev_is_ring = prev_col == 1 or prev_col == 6
        prev_is_middle = prev_col == 2 or prev_col == 5
        prev_is_pinky = prev_col == 0 or prev_col == 7

        mult = 0
        if (curr_is_index and prev_is_pinky and curr_below_prev) or (
            curr_is_pinky and prev_is_index and curr_above_prev
        ):
            mult = PINKY_ABOVE_INDEX if use_mult else 1
        elif (curr_is_ring and prev_is_middle and curr_above_prev) or (
            curr_is_middle and prev_is_ring and curr_below_prev
        ):
            mult = RING_ABOVE_MIDDLE if use_mult else 1
        elif (curr_is_pinky and prev_is_middle and curr_above_prev) or (
            curr_is_middle and prev_is_pinky and curr_below_prev
        ):
            mult = PINKY_ABOVE_MIDDLE if use_mult else 1
        elif (curr_is_pinky and prev_is_ring and curr_above_prev) or (
            curr_is_ring and prev_is_pinky and curr_below_prev
        ):
            mult = PINKY_ABOVE_RING if use_mult else 1

        # Penalize all outrolls that don't start on index finger
        if (
            use_mult
            and not prev_is_index
            and self.is_outward(0 if curr_is_left else 1, prev_col, curr_col)
        ):
            mult += OUTWARD

        return mult * freq / self.corpus_frequencies.bigrams_freq

    def evaluate_bigram_fast(self, bigram: str, freq: float):
        prev_char, curr_char = bigram[0], bigram[1]
        prev_col = self.column_dict.get(prev_char)
        curr_col = self.column_dict.get(curr_char)

        prev_row = self.row_dict.get(prev_char)
        curr_row = self.row_dict.get(curr_char)
        # Keys are on different hand or we haven't changed rows
        if curr_row == prev_row:
            return 0
        curr_above_prev = curr_row < prev_row
        curr_below_prev = curr_row > prev_row
        curr_is_pinky = curr_col == 0 or curr_col == 7
        curr_is_ring = curr_col == 1 or curr_col == 6
        curr_is_index = curr_col == 3 or curr_col == 4
        curr_is_middle = curr_col == 2 or curr_col == 5
        prev_is_index = prev_col == 3 or prev_col == 4
        prev_is_ring = prev_col == 1 or prev_col == 6
        prev_is_middle = prev_col == 2 or prev_col == 5
        prev_is_pinky = prev_col == 0 or prev_col == 7

        mult = 0
        if (curr_is_index and prev_is_pinky and curr_below_prev) or (
            curr_is_pinky and prev_is_index and curr_above_prev
        ):
            mult = PINKY_ABOVE_INDEX
        elif (curr_is_ring and prev_is_middle and curr_above_prev) or (
            curr_is_middle and prev_is_ring and curr_below_prev
        ):
            mult = RING_ABOVE_MIDDLE
        elif (curr_is_pinky and prev_is_middle and curr_above_prev) or (
            curr_is_middle and prev_is_pinky and curr_below_prev
        ):
            mult = PINKY_ABOVE_MIDDLE
        elif (curr_is_pinky and prev_is_ring and curr_above_prev) or (
            curr_is_ring and prev_is_pinky and curr_below_prev
        ):
            mult = PINKY_ABOVE_RING

        # Penalize all outrolls that don't start on index finger
        if not prev_is_index and self.is_outward(
            0 if curr_col <= 3 else 1, prev_col, curr_col
        ):
            mult += OUTWARD
        return mult * freq / self.corpus_frequencies.bigrams_freq

    def evaluate_fast(self, max=sys.float_info.max) -> float:
        res = 0
        for hand in self.keyboard.keyboard:
            for i, finger1 in enumerate(hand):
                for finger2 in hand[i + 1 :]:
                    for key1 in finger1:
                        for letter1 in key1.letters:
                            for key2 in finger2:
                                for letter2 in key2.letters:
                                    for bigram in [
                                        letter1 + letter2,
                                        letter2 + letter1,
                                    ]:
                                        freq = self.bigrams.get(bigram)
                                        if freq is None:
                                            continue
                                        res += self.evaluate_bigram_fast(bigram, freq)
                                        if res > max:
                                            return res
        return res

    def evaluate_bigram(self, bigram: tuple[str, float]) -> float:
        return self.evaluate_bigram_inner_old(bigram, True)

    def evaluate_bigram_stat(self, bigram: tuple[str, float]) -> float:
        return self.evaluate_bigram_inner_old(bigram, False)


def test_discomfort():
    kb = Keyboard(
        kb=[
            [
                [Key("aq"), Key("ir")],
                [Key("bs"), Key("jt")],
                [Key("cu"), Key("kv")],
                [Key("dw"), Key("lx")],
            ],
            [
                [Key("ey"), Key("mz")],
                [Key("f'"), Key("n")],
                [Key("g"), Key("o")],
                [Key("h."), Key("p,")],
            ],
        ],
        layout=[[2, 2, 2, 2], [2, 2, 2, 2]],
    )
    loops = 200
    discomfort_evaluator = DiscomfortEvaluator(kb)
    discomfort_sum1 = 0
    s = time.time()
    for _ in range(loops):
        discomfort_sum1 = 0
        for bigram in discomfort_evaluator.bigrams.items():
            discomfort_sum1 += discomfort_evaluator.evaluate_bigram(bigram)
    e = time.time()
    print(f"Old sum took {1000*(e-s)}ms")
    s = time.time()
    for _ in range(loops):
        discomfort_sum2 = discomfort_evaluator.evaluate_fast()
    e = time.time()
    print(f"New sum took {1000*(e-s)}ms")
    # print(discomfort_sum1)
    # print(discomfort_sum2)
    assert abs(discomfort_sum1 - discomfort_sum2) < 0.0001
    # TODO remove CorpusFrequencies and just bake it into all data
    # assert evaluator.evaluate_bigram(("ai", 1)) == 0
    # # Index key above ring
    # # assert evaluator.evaluate_bigram(("oe",1)) == 1
    # # Ring key above middle
    # assert evaluator.evaluate_bigram(("ng", 1)) == RING_ABOVE_MIDDLE
    # # Pinky key above middle
    # assert evaluator.evaluate_bigram(("ka", 1)) == PINKY_ABOVE_MIDDLE
    # # Pinky key above ring
    # assert evaluator.evaluate_bigram(("ja", 1)) == PINKY_ABOVE_RING
    # # Hand check works, ring on other hand
    # assert evaluator.evaluate_bigram(("na", 1)) == 0


# test_discomfort()
