from keyboard import Key, Keyboard
from custom_types import FreqList
from settings import GOAL_FINGER_MAX, PRINT
from words import CorpusFrequencies, create_inaccuracy_freq_list, get_characters


class FingerFreqEvaluator:
    frequencies: dict[str, float] = {}
    kb: Keyboard

    def __init__(self, freq_list: FreqList) -> None:
        self.set_letter_frequencies(freq_list)
        self.corpus_freq = CorpusFrequencies()

    def set_kb(self, k: Keyboard):
        self.kb = k

    def get_finger_frequencies(self) -> list[list[int]]:
        return [
            [
                sum(
                    self.frequencies[char]/self.corpus_freq.chars_freq if char in self.frequencies else 0
                    for key in col
                    for char in key.letters
                )
                for col in hand
            ]
            for hand in self.kb.keyboard
        ]

    def evaluate_finger_frequencies_MSE(self, goal: list[list[float]]) -> float:
        finger_frequencies = self.get_finger_frequencies()
        flattened_finger_frequencies = [
            ff for hand in finger_frequencies for ff in hand
        ]
        flattened_goal = [freq for hand in goal for freq in hand]
        return sum(
            (a - t) ** 2 for a, t in zip(flattened_finger_frequencies, flattened_goal)
        ) / len(finger_frequencies)

    def evaluate_finger_frequencies_MAPE(self, goal: list[list[float]]) -> float:
        finger_frequencies = self.get_finger_frequencies()
        flattened_finger_frequencies = [
            ff for hand in finger_frequencies for ff in hand
        ]
        flattened_goal = [freq for hand in goal for freq in hand]

        # MAPE calculation
        return sum(
            abs((a - t) / t)
            for a, t in zip(flattened_finger_frequencies, flattened_goal)
        ) / len(flattened_goal)

    def evaluate_finger_frequencies_max_limit_MAPE(self) -> float:
        finger_frequencies = self.get_finger_frequencies()
        flattened_finger_frequencies = [
            ff for hand in finger_frequencies for ff in hand
        ]
        flattened_goal = [freq for hand in GOAL_FINGER_MAX for freq in hand]

        return sum(
            (max(0, a - t) / t)
            for a, t in zip(flattened_finger_frequencies, flattened_goal)
        ) / len(flattened_goal)

    def set_letter_frequencies(self, freq_list: FreqList):
        self.frequencies = get_characters()


def test_evaluate_letter_freq():
    # Sum = 0.5+2*0.6+3*0.1 = 2
    # A freq = (0.5+0.6+0.1)/2 = 0.6
    # B freq = (0.1)/2 = 0.05
    # C freq = (0.1)/2 = 0.05
    # N freq = (0.6)/2 = 0.3
    # 0.3+0.05+0.05+0.6 = 1!
    ff = FingerFreqEvaluator(freq_list=[("a", 0.5), ("an", 0.6), ("cab", 0.1)])
    result = ff.frequencies
    epsilon = 1e-6
    assert abs(result["a"] - 0.6) < epsilon
    assert abs(result["b"] - 0.05) < epsilon
    assert abs(result["c"] - 0.05) < epsilon
    assert abs(result["n"] - 0.3) < epsilon
    kb: Keyboard = Keyboard(
        [
            [Key("ab"), Key("c"), Key("d"), Key("e"), Key("f")],
            [Key("g"), Key("h"), Key("i"), Key("j"), Key("k")],
        ]
    )
    ff.set_kb(kb)
    finger_freqs = ff.get_finger_frequencies()
    assert abs(finger_freqs[0][0] - 0.65) < epsilon
    assert abs(finger_freqs[0][1] - 0.05) < epsilon
    assert abs(finger_freqs[0][2] - 0.00) < epsilon
    goal_ff = [[0.65, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    assert ff.evaluate_finger_frequencies_MSE(goal_ff) - 0.0025 < epsilon

    print("test_evaluate_letter_freq passed!")


# test_evaluate_letter_freq()
