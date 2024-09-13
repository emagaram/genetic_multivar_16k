import sys
import time
from settings import SFB_SFS_DIFF_KEY_PENALTY, SFB_SFS_PINKY_PENALTY
from keyboard import Keyboard
from util import kb_to_column_dict, kb_to_row_dict, kb_to_reverse_column_dict, sort_str
from words import get_bigrams, get_skipgrams


class SFBSFSEvaluator:
    # column_dict = {}
    # row_dict = {}
    # reverse_col_dict: dict[int, str] = {}
    # bigrams: dict[str, float] = {}
    # skipgrams: list[dict[str, float]] = []
    # fast_bigrams: dict[str, float] = {}
    # fast_skipgrams: list[dict[str, float]] = []

    def __init__(self, kb: Keyboard = None) -> None:
        if kb:
            self.set_kb(kb)
        self.bigrams = get_bigrams()
        self.skipgrams = get_skipgrams()
        self.fast_bigrams:dict[str,float] = {}
        self.set_fast_bigrams(self.bigrams, self.fast_bigrams)
        self.set_fast_skipgrams()

    def set_kb(self, kb: Keyboard):
        self.column_dict = kb_to_column_dict(kb)
        self.reverse_column_dict = kb_to_reverse_column_dict(kb)
        self.row_dict = kb_to_row_dict(kb)

    def set_fast_bigrams(self, bigrams: dict[str, float], output: dict[str, float]):
        for bigram, freq in bigrams.items():
            sorted_bg = sort_str(bigram)
            output.setdefault(sorted_bg, 0)
            output[sorted_bg] += freq

    def set_fast_skipgrams(self):
        self.fast_skipgrams = [{} for _ in range(len(self.skipgrams))]
        for i, skipgrams in enumerate(self.skipgrams):
            self.set_fast_bigrams(skipgrams, self.fast_skipgrams[i])

    def evaluate_skipgrams_fast(
        self, skipgrams: dict[str, float], level: int, use_mult: bool
    ):
        return self.evaluate_bigrams_fast_inner(skipgrams, 0.5 ** (level + 1), use_mult)

    def evaluate_bigrams_fast(self, bigrams: dict[str, float], multiplier: float):
        return self.evaluate_bigrams_fast_inner(bigrams, multiplier, True)
    def evaluate_bigrams_fast_inner(
        self, bigrams: dict[str, float], multiplier: float, use_mult: bool
    ):
        sum = 0
        for finger, keys in self.reverse_column_dict.items():
            for i, letter1 in enumerate(keys):
                for letter2 in keys[i + 1 :]:
                    key = sort_str(letter1 + letter2)
                    freq = bigrams[key] if bigrams.get(key) != None else 0
                    multiplier_inner = multiplier
                    if use_mult:
                        if finger == 0 or finger == 7:
                            multiplier_inner *= SFB_SFS_PINKY_PENALTY
                        if self.row_dict.get(letter1) != self.row_dict.get(letter2):
                            multiplier_inner *= SFB_SFS_DIFF_KEY_PENALTY
                    sum += freq * multiplier_inner
        return sum

    def evaluate_bigram_inner(
        self, bigram: tuple[str, float], multiplier: float, use_mult: bool
    ) -> float:
        bigram_str, bigram_freq = bigram

        # We don't have to check this but it avoids adding a constant amount to all scores, so it saves time
        if bigram_str[1] == bigram_str[0]:
            return 0
        col_1 = self.column_dict.get(bigram_str[1])
        col_0 = self.column_dict.get(bigram_str[0])
        if col_1 == col_0 and col_0 != None:
            if use_mult:
                # Punish pinkies
                if col_1 == 0 or col_1 == 7:
                    multiplier *= SFB_SFS_PINKY_PENALTY
                # Punish distance
                if self.row_dict.get(bigram_str[1]) != self.row_dict.get(bigram_str[0]):
                    multiplier *= SFB_SFS_DIFF_KEY_PENALTY
            return multiplier * bigram_freq
        return 0

    def evaluate_bigram(self, bigram: tuple[str, float]) -> float:
        return self.evaluate_bigram_inner(bigram, 1, True)

    def evaluate_bigram_stat(self, bigram: tuple[str, float]) -> float:
        return self.evaluate_bigram_inner(bigram, 1, False)

    def evaluate_skipgram(self, skipgram: tuple[str, float, int]) -> float:
        skipgram_str, skipgram_frec, skipgram_level = skipgram
        return self.evaluate_bigram_inner(
            (skipgram_str, skipgram_frec), 0.5 ** (skipgram_level + 1), True
        )

    def evaluate_skipgram_stat(self, skipgram: tuple[str, float]) -> float:
        return self.evaluate_bigram_inner(skipgram, 1, False)


def test_speed():
    kb: Keyboard = Keyboard(
        [
            [
                ["a", "s"],
                ["d", "f"],
            ],  # First hand: two columns ('a', 's') in column 0, ('d', 'f') in column 1
            [
                ["g", "h"],
                ["j", "k"],
            ],  # Second hand: two columns ('g', 'h') in column 0, ('j', 'k') in column 1
        ]
    )
    evaluator = SFBSFSEvaluator(kb)
    sfb = 0
    sfb_start = time.time()
    for bigram, freq in evaluator.bigrams.items():
        sfb += evaluator.evaluate_bigram((bigram, freq))
    sfb_end = time.time()
    sfb_fast_start = time.time()
    sfb_fast = evaluator.evaluate_bigrams_fast_inner(evaluator.fast_bigrams, 1, True)
    sfb_fast_end = time.time()
    sfs = 0
    sfs_start = time.time()
    for i, skipgrams in enumerate(evaluator.skipgrams):
        for skip_str, skip_freq in skipgrams.items():
            sfs += evaluator.evaluate_skipgram((skip_str, skip_freq, i))    
    sfs_end = time.time()
    sfs_fast = 0
    sfs_fast_start = time.time()
    for i, skipgrams in enumerate(evaluator.fast_skipgrams):
        sfs_fast+=evaluator.evaluate_skipgrams_fast(skipgrams, i, True)
    sfs_fast_end = time.time()
    
    # print(sfs_fast)
    assert abs(sfb_fast - sfb) < sys.float_info.epsilon
    assert abs(sfs_fast - sfs) < sys.float_info.epsilon
    # print(f"Regular sfb took {1000*(sfb_end-sfb_start):.3f}")
    # print(f"Fast sfb took {1000*(sfb_fast_end-sfb_fast_start):.3f}")
    # print(f"Regular sfs took {1000*(sfs_end-sfs_start):.3f}")
    # print(f"Fast sfs took {1000*(sfs_fast_end-sfs_fast_start):.3f}")    
    


def test_evaluate_sfb():
    # Define a sample keyboard layout
    kb1: Keyboard = Keyboard(
        [
            [
                ["a", "s"],
                ["d", "f"],
            ],  # First hand: two columns ('a', 's') in column 0, ('d', 'f') in column 1
            [
                ["g", "h"],
                ["j", "k"],
            ],  # Second hand: two columns ('g', 'h') in column 0, ('j', 'k') in column 1
        ]
    )
    grams = {
        "as": 0.05,
        "df": 0.10,
        "ah": 0.20,
        "dj": 0.15,
        "gk": 0.12,
        "gh": 0.25,
    }
    res_sfb = 0
    res_sfs = 0
    evaluator1 = SFBSFSEvaluator(kb1)
    for bigram, freq in grams.items():
        res_sfb += evaluator1.evaluate_bigram((bigram, freq))
        res_sfs += evaluator1.evaluate_skipgram((bigram, freq, 0))
        res_sfs += evaluator1.evaluate_skipgram((bigram, freq, 1))

    res_sfb_fast = evaluator1.evaluate_bigrams_fast_inner(grams, 1, True)
    res_sfs_fast = evaluator1.evaluate_skipgrams_fast(
        grams, 0, True
    ) + evaluator1.evaluate_skipgrams_fast(grams, 1, True)

    # print(res_sfb_fast)
    # print(res_sfb)
    assert res_sfb_fast == res_sfb
    assert res_sfs_fast == res_sfs
    # print("test_evaluate_sfb passed!")


# Run the test
test_evaluate_sfb()
test_speed()
