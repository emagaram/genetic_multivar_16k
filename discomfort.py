from keyboard import Keyboard, RandomKeyboard
from util import kb_to_column_dict, kb_to_row_dict


class HindranceEvaluator:
    column_dict = {}
    row_dict = {}

    def __init__(self, kb: RandomKeyboard = None) -> None:
        if kb:
            self.set_kb(kb)

    def set_kb(self, kb: RandomKeyboard):
        self.column_dict = kb_to_column_dict(kb)
        self.row_dict = kb_to_row_dict(kb)

    def evaluate_bigram(self, bigram: tuple[str, float]) -> float:
        """
        * 1: Pinky types key above index*
        * 1: Index types key above ring*
        * 1.5: Ring types key above middle
        * 3: Index types key above middle*
        * 4: Pinky types key above middle
        * 4.5: Pinky types key above ring

        """
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

        # Index key above ring
        if (curr_is_index and prev_is_ring and curr_above_prev) or (curr_is_ring and prev_is_index and curr_below_prev):
            return 1*freq
        # Ring key above middle
        if (curr_is_ring and prev_is_middle and curr_above_prev) or (curr_is_middle and prev_is_ring and curr_below_prev):
            return 1.5*freq
        # Pinky types key above middle
        if (curr_is_pinky and prev_is_middle and curr_above_prev) or (curr_is_middle and prev_is_pinky and curr_below_prev):
            return 4*freq
        # Pinky key above ring
        if (curr_is_pinky and prev_is_ring and curr_above_prev) or (curr_is_ring and prev_is_pinky and curr_below_prev):
            return 4.5*freq
        return 0


def test_hindrance():
    kb = Keyboard([
        [["a", "i"], ["b", "j"], ["c", "k"], ["d", "l"]],
        [["e", "m"], ["f", "n"], ["g", "o"], ["h", "p"]],
    ])
    evaluator = HindranceEvaluator(kb)
    assert evaluator.evaluate_bigram(("ai",1)) == 0
    # Index key above ring
    assert evaluator.evaluate_bigram(("oe",1)) == 1
    # Ring key above middle
    assert evaluator.evaluate_bigram(("ng",1)) == 1.5
    # Pinky key above middle
    assert evaluator.evaluate_bigram(("ka",1)) == 4
    # Pinky key above ring
    assert evaluator.evaluate_bigram(("ja",1)) == 4.5
    # Hand check works, ring on other hand
    assert evaluator.evaluate_bigram(("na",1)) == 0
    # print("Hindrance tests passed!")


test_hindrance()
