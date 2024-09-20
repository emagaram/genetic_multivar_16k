import unittest
from keyboard import Key, Keyboard
from custom_types import FreqList
from settings import EFFORT_BEAKLISH, FINGER_MAX_GRID, PRINT
from words import CorpusFrequencies


class EffortEvaluator:
    frequencies: dict[str, float] = {}
    kb: Keyboard

    def __init__(
        self, effort_grid: list[list[list[int]]], char_frequencies: dict[str, float]
    ) -> None:
        self.frequencies = char_frequencies
        self.corpus_freq = CorpusFrequencies()
        self.effort_grid = effort_grid

    def set_kb(self, k: Keyboard):
        self.kb = k

    def evaluate_effort(self):
        return sum(
            sum(self.frequencies[char] * self.effort_grid[i][j][k] for char in key.letters)
            for i, hand in enumerate(self.kb.keyboard)
            for j, finger in enumerate(hand)
            for k, key in enumerate(finger)
        ) / self.corpus_freq.chars_freq


class TestEffortEvaluator(unittest.TestCase):
    def test_evaluate_effort(self):
        # Set up the keyboard with known keys and positions
        # Keyboard structure: kb.keyboard[hand_index][finger_index][key_index]
        effort = [[[2, 0], [1, 0]], [[0, 1], [1, 0]]]
        frequencies = {
            "a": 2,
            "b": 1,
            "c": 1.6,
            "d": 1,
        }
        layout = [[2, 2], [2, 2]]
        kb = [
            [  # Left hand
                [Key("a"), Key("b")],  # Finger 0
                [Key("c"), Key("d")],  # Finger 1
            ],
            [  # Right hand
                [Key("a"), Key("c")],  # Finger 0
                [Key("b"), Key("d")],  # Finger 1
            ],
        ]
        
        keyboard = Keyboard(layout, kb)
        print(keyboard.str_display())
        # Initialize EffortEvaluator and set the keyboard
        evaluator = EffortEvaluator(effort, frequencies)
        evaluator.set_kb(keyboard)


        expected_effort = 8.2/evaluator.corpus_freq.chars_freq
        # Evaluate effort using the method
        calculated_effort = evaluator.evaluate_effort()

        # Assert that the calculated effort matches the expected effort
        self.assertAlmostEqual(calculated_effort, expected_effort, places=6)

        # Print the result (optional)
        print(f"Expected Effort: {expected_effort}")
        print(f"Calculated Effort: {calculated_effort}")


# Run the test
if __name__ == "__main__":
    unittest.main()
