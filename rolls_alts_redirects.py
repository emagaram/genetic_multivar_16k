import sys
import time
from redirects import RedirectsEvaluator
from rolls import RollEvaluator
from settings import ADJACENT_BONUS, BAD_REDIRECT, INWARD_BONUS
from keyboard import Key, Keyboard
from sfb_sfs import SFBSFSEvaluator
from util import kb_to_column_dict, kb_to_reverse_column_dict, kb_to_row_dict
from words import CorpusFrequencies, get_trigrams


# Rolls Alts Redirects
class RAREvaluator:

    def __init__(self, kb: Keyboard = None) -> None:
        if kb:
            self.set_kb(kb)
        self.trigrams = get_trigrams()
        self.corpus_frequencies = CorpusFrequencies()

    def set_kb(self, kb: Keyboard):
        self.column_dict = kb_to_column_dict(kb)
        self.reverse_column_dict = kb_to_reverse_column_dict(kb)
        self.row_dict = kb_to_row_dict(kb)

    def get_hand_pos(self, trigram: str) -> tuple[int, int, int]:
        res = []
        for char in trigram:
            res.append(1 if self.column_dict[char] > 3 else 0)
        return tuple(res)

    def evaluate_rolls_alts_redirects_stat(self) -> tuple[float, float, float]:
        alts = 0
        rolls = 0
        redirects = 0
        for trigram, freq in self.trigrams.items():
            hand_pos = self.get_hand_pos(trigram)
            char1, char2, char3 = trigram[0], trigram[1], trigram[2]
            col1, col2, col3 = (
                self.column_dict[char1],
                self.column_dict[char2],
                self.column_dict[char3],
            )
            if not (col1 != col2 != col3):
                continue
            if hand_pos == (0, 1, 0) or hand_pos == (1, 0, 1):
                alts += freq
            elif (
                hand_pos == (0, 0, 1)
                or hand_pos == (0, 1, 1)
                or hand_pos == (1, 0, 0)
                or hand_pos == (1, 1, 0)
            ):
                rolls += freq
            elif hand_pos == (0, 0, 0) or hand_pos == (1, 1, 1):
                if (col1 < col2 > col3) or (col1 > col2 < col3):
                    redirects += freq

        return (
            rolls / self.corpus_frequencies.trigrams_freq,
            alts / self.corpus_frequencies.trigrams_freq,
            redirects / self.corpus_frequencies.trigrams_freq,
        )

    # Put this outside so it can be tested
    def is_same_row_or_longer_higher(
        self, hand: int, col1: int, col2: int, row1: int, row2: int
    ):
        height = [[0, 1, 2, 1], [1, 2, 1, 0]]
        c1_height = height[hand][col1]
        c2_height = height[hand][col2]
        row_diff = row2 - row1
        height_diff = c2_height - c1_height
        return row1 == row2 or (-row_diff > 0) == (height_diff > 0)

    def evaluate_rolls_redirects(self) -> tuple[float, float]:

        def is_pinky(hand: int, col: int):
            return (hand == 0 and col == 0) or (hand == 1 and col == 3)

        def is_ring(hand: int, col: int):
            return (hand == 0 and col == 1) or (hand == 1 and col == 2)

        def is_adjacent(col1: int, col2: int):
            return abs(col2 - col1) == 1

        def is_inward(hand: int, i1: int, i2: int):
            return (hand == 1 and i1 > i2) or (hand == 0 and i2 > i1)

        def is_index(hand: int, col: int):
            return (col == 3 and hand == 0) or (col == 0 and hand == 1)

        rolls = 0
        redirects = 0
        for trigram, freq in self.trigrams.items():
            hand_pos = self.get_hand_pos(trigram)
            char1, char2, char3 = trigram[0], trigram[1], trigram[2]
            col1, col2, col3 = (
                self.column_dict[char1],
                self.column_dict[char2],
                self.column_dict[char3],
            )
            if not (col1 != col2 != col3):
                continue
            if (
                hand_pos == (0, 0, 1)
                or hand_pos == (0, 1, 1)
                or hand_pos == (1, 0, 0)
                or hand_pos == (1, 1, 0)
            ):
                rolling_hand = hand_pos[1]
                first_half_roll = hand_pos == (0, 0, 1) or hand_pos == (1, 1, 0)
                i1, i2 = (
                    (self.column_dict[char1], self.column_dict[char2])
                    if first_half_roll
                    else (self.column_dict[char2], self.column_dict[char3])
                )

                c1, c2 = (char1, char2) if first_half_roll else (char2, char3)
                row1 = self.row_dict[c1]
                row2 = self.row_dict[c2]
                if is_pinky(rolling_hand, i1 % 4):
                    continue
                if is_ring(rolling_hand, i1 % 4) and is_pinky(rolling_hand, i2 % 4):
                    continue
                if not self.is_same_row_or_longer_higher(
                    rolling_hand, i1 % 4, i2 % 4, row1, row2
                ):
                    continue
                add = freq
                if is_inward(rolling_hand, i1 % 4, i2 % 4):
                    add *= INWARD_BONUS
                if is_adjacent(i1, i2):
                    add *= ADJACENT_BONUS
                rolls += add
            elif hand_pos == (1, 1, 1) or hand_pos == (0, 0, 0):
                hand = hand_pos[0]
                if (col1 < col2 > col3) or (col1 > col2 < col3):
                    if (
                        is_index(hand, col1 % 4)
                        or is_index(hand, col2 % 4)
                        or is_index(hand, col3 % 4)
                    ):
                        redirects += freq
                    else:
                        redirects += freq * BAD_REDIRECT
        return (
            rolls / self.corpus_frequencies.trigrams_freq,
            redirects / self.corpus_frequencies.trigrams_freq,
        )


def test_typing():
    keyboard_bad: Keyboard = Keyboard(
        [[2, 2, 2, 2], [2, 2, 2, 2]],
        # . hu   , ef   'ox az   ik py   |   n r   cl gm   dw qt   bv js
        kb=[
            [
                [Key("aq"), Key("br")],
                [Key("cs"), Key("dt")],
                [Key("eu"), Key("fv")],
                [Key("gw"), Key("hx")],
            ],
            [
                [Key("iy"), Key("jz")],
                [Key("k'"), Key("l")],
                [Key("m"), Key("n")],
                [Key("o."), Key("p,")],
            ],
        ],
    )
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
    kb = keyboard
    typing_evaluator = RAREvaluator(kb)
    rolls_eval = RollEvaluator(kb)
    redirect_eval = RedirectsEvaluator(kb)
    loops = 200
    typing_rolls_redirects_start = time.time()
    for _ in range(loops):
        typing_rolls, typing_redirects = typing_evaluator.evaluate_rolls_redirects()
    typing_rolls_redirects_end = time.time()
    print(f"Evaluating typing rolls and redirects {loops}x took {1000*(typing_rolls_redirects_end-typing_rolls_redirects_start)}ms")
    rolls_rolls = rolls_eval.evaluate_fast()

    redirect_rolls_start = time.time()
    for _ in range(loops):
        redirect_redirects = redirect_eval.evaluate_fast()
        rolls = rolls_eval.evaluate_fast()
    redirect_rolls_end = time.time()
    print(f"Evaluating rolls and redirects {loops}x took {1000*(redirect_rolls_end-redirect_rolls_start)}ms")
    rolls_all_rolls = rolls_eval.evaluate_fast_stat()
    
    redirect_redirects_stat = sum(redirect_eval.evaluate_trigram_stat(trigram) for trigram in redirect_eval.trigrams.items())
    typing_all_rolls, alts_all, redirect_all = typing_evaluator.evaluate_rolls_alts_redirects_stat()
    
    assert abs(typing_redirects - redirect_redirects) < 0.00001
    assert abs(typing_rolls - rolls_rolls) < 0.00001
    assert abs(typing_all_rolls - rolls_all_rolls) < 0.00001
    assert abs(redirect_redirects_stat - redirect_all) < 0.00001


# test_typing()
