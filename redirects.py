import sys
import time
from settings import BAD_REDIRECT, NORMAL_REDIRECT, PRINT
from keyboard import Key, Keyboard
from util import kb_to_column_dict, kb_to_reverse_column_dict
from words import CorpusFrequencies, get_trigrams


class RedirectsEvaluator:
    def generate_redirect_indexes(n):
        result = []
        for i in range(n + 1):
            for j in range(n + 1):
                for k in range(n + 1):
                    if ((i < j > k) or (i > j < k)) and (i != j != k):
                        result.append((i, j, k))
        return result

    POSITIONS_TO_EVAL: list[tuple[int, int, int]] = generate_redirect_indexes(3)

    def __init__(self, kb: Keyboard = None) -> None:
        if kb:
            self.set_kb(kb)
        self.trigrams = get_trigrams()
        self.corpus_frequencies = CorpusFrequencies()

    def set_kb(self, kb: Keyboard):
        self.column_dict = kb_to_column_dict(kb)
        self.reverse_column_dict = kb_to_reverse_column_dict(kb)

    def evaluate_fast(self, max: float = sys.float_info.max) -> float:
        return self.evaluate_fast_inner(True, max)

    def evaluate_fast_inner(self, use_mult: bool, max=sys.float_info.max) -> float:
        hand_len = 4
        reverse_column_dict = [
            [self.reverse_column_dict[i] for i in range(hand_len)],
            [self.reverse_column_dict[i] for i in range(hand_len, 2 * hand_len)],
        ]

        def is_index(hand: int, col: int):
            return (col == 3 and hand == 0) or (col == 0 and hand == 1)

        score = 0
        for hand_i, hand in enumerate(reverse_column_dict):
            for i, j, k in RedirectsEvaluator.POSITIONS_TO_EVAL:
                for char1 in hand[i]:
                    for char2 in hand[j]:
                        for char3 in hand[k]:
                            freq = self.trigrams.get(char1 + char2 + char3)
                            if freq != None:
                                if (
                                    not use_mult
                                    or is_index(hand_i, i)
                                    or is_index(hand_i, j)
                                    or is_index(hand_i, k)
                                ):
                                    score += NORMAL_REDIRECT * freq
                                else:
                                    score += freq * BAD_REDIRECT
                                if score > max:
                                    return score
        return score / self.corpus_frequencies.trigrams_freq

    def evaluate_trigram_inner(self, trigram: tuple[str, float], use_mult: bool):
        def on_same_hand(col0: int, col1: int, col2: int):
            return (col0 <= 3 and col1 <= 3 and col2 <= 3) or (
                col0 > 3 and col1 > 3 and col2 > 3
            )

        def is_index_finger(col: int):
            return col == 3 or col == 4

        col0, col1, col2 = (self.column_dict.get(char) for char in trigram[0])
        if col0 is None or col1 is None or col2 is None:
            return 0
        if not on_same_hand(col0, col1, col2):
            return 0
        # Right and then left, left and then right
        if (col1 > col0 and col1 > col2) or (col1 < col0 and col2 > col1):
            if not use_mult or (
                is_index_finger(col0) or is_index_finger(col1) or is_index_finger(col2)
            ):
                return (
                    trigram[1] * NORMAL_REDIRECT / self.corpus_frequencies.trigrams_freq
                )
            else:
                return trigram[1] * BAD_REDIRECT / self.corpus_frequencies.trigrams_freq
        return 0

    def evaluate_trigram(self, trigram: tuple[str, float]):
        return self.evaluate_trigram_inner(trigram, True)

    def evaluate_trigram_stat(self, trigram: tuple[str, float]):
        return self.evaluate_trigram_inner(trigram, False)


def test_redirect():
    keyboard: Keyboard = Keyboard(
        [[2, 2, 2, 2], [2, 2, 2, 2]],
        kb=[
            [
                [Key("a"), Key("b")],
                [Key("c"), Key("d")],
                [Key("e"), Key("f")],
                [Key("g"), Key("h")],
            ],
            [
                [Key("i"), Key("j")],
                [Key("k"), Key("l")],
                [Key("m"), Key("n")],
                [Key("o"), Key("p")],
            ],
        ],
    )
    redirect_eval = RedirectsEvaluator(keyboard)
    # TODO remove CorpusFrequencies and just bake it into all data
    # assert redirect_eval.evaluate_trigram(("aca", 2)) == 2 * BAD_REDIRECT
    # assert redirect_eval.evaluate_trigram(("acc", 2)) == 0
    # assert redirect_eval.evaluate_trigram(("acd", 2)) == 0
    # assert redirect_eval.evaluate_trigram(("ecf", 2)) == 2 * BAD_REDIRECT
    # assert redirect_eval.evaluate_trigram(("egi", 2)) == 0
    # assert redirect_eval.evaluate_trigram(("ggg", 2)) == 0
    # assert redirect_eval.evaluate_trigram(("agh", 2)) == 0
    # assert redirect_eval.evaluate_trigram(("agi", 2)) == 0
    # assert redirect_eval.evaluate_trigram(("gag", 2)) == 2

    random_kb = Keyboard([[2, 2, 2, 2], [2, 2, 2, 2]])
    redirect_eval = RedirectsEvaluator(random_kb)
    start_fast = time.time()
    fast = redirect_eval.evaluate_fast()
    end_fast = time.time()
    slow = 0
    start_slow = time.time()
    for trigram in redirect_eval.trigrams.items():
        slow += redirect_eval.evaluate_trigram(trigram)
    end_slow = time.time()
    assert abs(slow - fast) < 0.000001
    print("Redirect tests passed!")


# test_redirect()
