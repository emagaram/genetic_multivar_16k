import random
from constants_weight import USE_PUNCTUATION
from words import get_letters, get_punctuation

PunctuationOption = tuple[tuple[int, int, int], tuple[int, int, int]]


class Keyboard:
    keyboard: list[list[list[str]]]

    def __init__(self, kb: list[list[list[str]]]) -> None:
        self.keyboard = kb

    def __str__(self):
        res = ""
        space = "   "
        for i, hand in enumerate(self.keyboard):
            for j, col in enumerate(hand):
                for k, key in enumerate(col):
                    key = "".join(sorted(key))
                    res += key
                    res += " " if k != len(col) - 1 else ""
                res += space if j != len(hand) - 1 else ""
            res += f"{space}|{space}" if i != len(self.keyboard) - 1 else ""
        return res

    def __eq__(self, other):
        if not isinstance(other, Keyboard):
            return False
        for i, hand in enumerate(self.keyboard):
            for j, col in enumerate(hand):
                for k in range(len(col)):
                    if self.keyboard[i][j][k] != other.keyboard[i][j][k]:
                        return False
        return True

    def __hash__(self):
        return hash(self.__str__())


PUNCTUATION_OPS_2_SYMS: list[PunctuationOption] = [
    # Right bottom
    ((1, 2, 1), (1, 3, 1)),
    # Left bottom
    ((0, 0, 1), (0, 1, 1)),
    # Left right bottom
    ((0, 0, 1), (1, 3, 1)),
    # Right top
    ((1, 2, 0), (1, 3, 0)),
    # Left top
    ((0, 0, 0), (0, 1, 0)),
    # Left right top
    ((0, 0, 0), (1, 3, 0)),
]


class RandomKeyboard(Keyboard):
    def __init__(self, layout: list[list[int]]) -> None:
        self.generate_random_keyboard(layout)

    def generate_random_keyboard(self, layout: list[list[int]]):
        letters = get_letters()
        punctuation = get_punctuation()
        random.shuffle(letters)
        random.shuffle(punctuation)
        punctuation_option = random.choice(PUNCTUATION_OPS_2_SYMS)

        self.keyboard = [
            [
                [
                    (
                        letters.pop()
                        if not USE_PUNCTUATION or (i, j, k) not in punctuation_option
                        else punctuation.pop()
                    )
                    for k in range(column)
                ]
                for j, column in enumerate(hand)
            ]
            for i, hand in enumerate(layout)
        ]

        for letter in letters:
            col, key, _ = self.get_random_non_punc_kb_index()
            col[key] += letter

    def get_random_non_punc_kb_index(self) -> tuple[list[str], int, int]:
        punctuation_set = set(get_punctuation())
        while True:
            rand_hand = random.choice(self.keyboard)
            rand_col = random.choice(rand_hand)
            rand_key_idx = random.randrange(len(rand_col))
            rand_key = rand_col[rand_key_idx]
            if all(char not in punctuation_set for char in rand_key):
                break
        if len(rand_key) > 0:
            rand_letter_idx = random.randrange(len(rand_key))
            return (rand_col, rand_key_idx, rand_letter_idx)
        else:
            raise Exception("Random key doesn't have any letters.")


def test_random_keyboard():
    layout = [[2, 2, 2, 2], [2, 2, 2, 2]]
    kb = RandomKeyboard(layout)
    # Manually check it looks right
    print(str(kb))


# test_random_keyboard()
