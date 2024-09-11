from keyboard import Keyboard
from util import kb_to_column_dict
from words import get_trigrams


class RedirectEvaluator:
    column_dict = {}
    trigrams: dict[str, float] = {}

    def __init__(self, kb: Keyboard = None) -> None:
        if kb:
            self.set_kb(kb)
        self.trigrams = get_trigrams()

    def set_kb(self, kb: Keyboard):
        self.column_dict = kb_to_column_dict(kb)

    def evaluate_trigram(self, trigram: tuple[str, float]):
        def on_same_hand(col0: int, col1: int, col2: int):
            return (col0 <= 3 and col1 <= 3 and col2 <= 3) or (
                col0 > 3 and col1 > 3 and col2 > 3
            )

        col0, col1, col2 = (self.column_dict[char] for char in trigram[0])
        if not on_same_hand(col0, col1, col2):
            return 0
        # Right and then left, left and then right
        if (col1 > col0 and col1 > col2) or (col1 < col0 and col2 > col1):
            return trigram[1]
        return 0


def test_redirect():
    keyboard: Keyboard = Keyboard(
        [
            [["a", "b"], ["c", "d"], ["e", "f"], ["g", "h"]],
            [["i", "j"], ["k", "l"], ["m", "n"], ["o", "p"]],
        ]
    )
    redirect_eval = RedirectEvaluator(keyboard)
    assert redirect_eval.evaluate_trigram(("aca", 2)) == 2
    assert redirect_eval.evaluate_trigram(("acc", 2)) == 0
    assert redirect_eval.evaluate_trigram(("acd", 2)) == 0
    assert redirect_eval.evaluate_trigram(("ecf", 2)) == 2
    assert redirect_eval.evaluate_trigram(("egi", 2)) == 0
    assert redirect_eval.evaluate_trigram(("ggg", 2)) == 0
    assert redirect_eval.evaluate_trigram(("agh", 2)) == 0
    assert redirect_eval.evaluate_trigram(("agi", 2)) == 0
    assert redirect_eval.evaluate_trigram(("gag", 2)) == 2
    print("Redirect tests passed!")


# test_redirect()
