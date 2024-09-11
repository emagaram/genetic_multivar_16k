import constants
from keyboard import Keyboard
from util import kb_to_column_dict, kb_to_row_dict
from words import get_bigrams, get_skipgrams

class SFBSFSEvaluator:
    column_dict = {}
    row_dict = {}
    bigrams = {}
    skipgrams = []
    def __init__(self, kb: Keyboard = None) -> None:
        if kb:
            self.set_kb(kb)
        self.bigrams = get_bigrams()
        self.skipgrams = get_skipgrams()

    def set_kb(self, kb: Keyboard):
        self.column_dict = kb_to_column_dict(kb)
        self.row_dict = kb_to_row_dict(kb)

    def evaluate_bigram_inner(self, bigram: tuple[str, float], multiplier:float) -> float:
        bigram_str, bigram_freq = bigram
        if(bigram_str[1] == bigram_str[0]):
            return 0
        col_1 = self.column_dict.get(bigram_str[1])
        if (
            col_1 == self.column_dict.get(bigram_str[0])
            and self.column_dict.get(bigram_str[0]) != None
        ):
            if col_1 == 0 or col_1 == 7:
                multiplier*=constants.SFB_SFS_PINKY_PUNISHMENT
            if self.row_dict.get(bigram_str[1]) == self.row_dict.get(bigram_str[0]):  # Same key
                return multiplier * bigram_freq
            else:
                return constants.SFB_SFS_DIFF_KEY_PENALTY * multiplier * bigram_freq
        return 0

    def evaluate_bigram(self, bigram: tuple[str, float]) -> float:
        return self.evaluate_bigram_inner(bigram, 1)

    def evaluate_skipgram(self, skipgram: tuple[str, float, int]) -> float:
        skipgram_str, skipgram_frec, skipgram_level = skipgram
        return self.evaluate_bigram_inner(
            (skipgram_str, skipgram_frec), 0.5 ** (skipgram_level + 1)
        )

3*3*0.5



def test_evaluate_sfb():
    # Define a sample keyboard layout
    kb:Keyboard = Keyboard([
        [
            ["a", "s"],
            ["d", "f"],
        ],  # First hand: two columns ('a', 's') in column 0, ('d', 'f') in column 1
        [
            ["g", "h"],
            ["j", "k"],
        ],  # Second hand: two columns ('g', 'h') in column 0, ('j', 'k') in column 1
    ])
    # Define a list of words and their frequencies

    # Expected result is the sum of frequencies where consecutive characters are in the same column
    
    # 'as' => 0.05*DIFF_KEY_PENALTY*PINKY_PUNISHMENT, 'df' => 0.10*DIFF_KEY_PENALTY, 'gh' => 0.25*DIFF_KEY_PENALTY
    # as = 3*3*0.05 + 3*3*0.05*0.5 + 3*3*0.05*0.5^2
    # df = 3*0.1 + 3*0.1*0.5 + 3*0.1*0.5^2
    # gh = 3*0.25*0.5 + 3*0.25*0.5^2
    
    
    grams = {
        "as": 0.05,
        "df": 0.10,
        "ah": 0.20,
        "dj": 0.15,
        "gk": 0.12,
        "gh": 0.25,
    }
    expected_result = 2.625
    res = 0
    evaluator = SFBSFSEvaluator(kb)
    
    for bigram, freq in grams.items():    
        res+=evaluator.evaluate_bigram((bigram, freq))
        # print(res)
        res+=evaluator.evaluate_skipgram((bigram, freq, 0))
        res+=evaluator.evaluate_skipgram((bigram, freq, 1))
    assert (
        abs(res - expected_result) < 1e-6
    ), f"Expected {expected_result}, but got {res}"
    print("test_evaluate_sfb passed!")


# Run the test
# test_evaluate_sfb()
