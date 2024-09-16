import random
import string
from custom_types import KeyboardLocation, PunctuationOption
from settings import USE_PUNCTUATION
import settings
from words import get_letters, get_punctuation

def get_t10_keys():
    return list(string.ascii_uppercase)        


class Key:
    def __init__(self,letters:str):
        self.letters = letters 


PUNCTUATION_OPTIONS_2_SYMS: list[PunctuationOption] = [
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
    ((0, 0, 0), (1, 3, 0))
]

MAGIC_OPTIONS: list[KeyboardLocation] = [
    # Left index top
    (0, 3, 0),
    # Left middle top
    (0, 2, 0),
    # Right index top
    (1, 0, 0),
    # Right middle top
    (1, 1, 0)
]

class MagicKey(Key):
    MAGIC_LETTER = "*"
    def __init__ (self, layout: list[list[int]]):
        super().__init__(MagicKey.MAGIC_LETTER)
        self.uses:dict[str,str] = self.get_random_uses(layout)
        
    def get_random_uses(self, layout: list[list[int]]) -> dict[str,str]:
        t10_keys = get_t10_keys()
        index = 0
        uses = {}
        letters = get_letters()
        random.shuffle(letters)
        for hand in layout:
           for col in hand:
               for _ in range(col):
                   uses[t10_keys[index]] = random.choice(letters)
                   index+=1
        return uses
    
    def mutate(self):
        rand_key = random.choice(list(self.uses.keys()))
        self.uses[rand_key] = random.choice(get_letters())  
              
        
class Keyboard():         

    
    # [ [[ab,cd],[ef,gh]], [[hi,jk],[lm,no]] ]
    # "h", goal 7 => num_rows*col idx = 2*3 = 6 + rows_idx = 1 => 6 + 1
    punctuation_set = set(get_punctuation())    
    non_letters_set = set(get_punctuation()+[MagicKey.MAGIC_LETTER])    
    # def get_flattened_idx(self, hand_idx:int, col_idx:int, row_idx:int):
    #     if self.keyboard == []:
    #         raise Exception("Empty Keyboard")
    #     cols_per_hand = len(self.keyboard[0])
    #     num_rows = len(self.keyboard[0][0])
    #     return hand_idx*cols_per_hand*num_rows + col_idx*num_rows + row_idx

    # def get_unflattened_idx(self, idx:int) -> KeyboardLocation:
    #     if self.keyboard == []:
    #         raise Exception("Empty Keyboard")
    #     cols_per_hand = len(self.keyboard[0])
    #     num_rows = len(self.keyboard[0][0])
    #     hand_idx = idx % (num_rows*cols_per_hand)
    #     idx -= hand_idx*num_rows*cols_per_hand
    #     row_idx = idx % num_rows
    #     pass
    def get_punctuation_locations(self) -> PunctuationOption:
        result = []
        for i, hand in enumerate(self.keyboard):
            for j, col in enumerate(hand):
                for k, key in enumerate(col):
                    if key.letters[0] in self.punctuation_set:
                        result.append((i, j, k))


        return tuple(result)
    def __init__(self, layout: list[list[int]], kb: list[list[list[Key]]] = []) -> None:
        self.punctuation_location = (
            random.choice(PUNCTUATION_OPTIONS_2_SYMS) if USE_PUNCTUATION else []
        )
        self.magic_locations:list[KeyboardLocation] = []
        while len(self.magic_locations) < settings.NUM_MAGIC:
            new_location = random.choice(MAGIC_OPTIONS)
            if (
                new_location != self.punctuation_location[0]
                and new_location != self.punctuation_location[1]
            ):
                self.magic_locations.append(new_location)
        if kb == []:
            self.generate_random_keyboard(layout)
        else:
            self.keyboard = kb
            self.punctuation_location = self.get_punctuation_locations()
        
    def magic_keys_str(self, space = " "):
        res = ""
        for i, (hand, col, row) in enumerate(self.magic_locations):
            mk: MagicKey = self.keyboard[hand][col][row]
            res+=f"Magic Key {i}\n"
            for use in mk.uses.items():
                res+=f"{space}{use[0]}:{use[1]}\n"        
        return res
    def __str__(self):
        res = ""
        space = "   "
        for i, hand in enumerate(self.keyboard):
            for j, col in enumerate(hand):
                for k, key in enumerate(col):
                    key = "".join(sorted(key.letters))
                    res += key
                    res += " " if k != len(col) - 1 else ""
                res += space if j != len(hand) - 1 else ""
            res += f"{space}|{space}" if i != len(self.keyboard) - 1 else ""
        res+="\nMagic:\n"
        res+=self.magic_keys_str()
        return res

    def __eq__(self, other):
        if not isinstance(other, Keyboard):
            return False
        for i, hand in enumerate(self.keyboard):
            for j, col in enumerate(hand):
                for k in range(len(col)):
                    if self.keyboard[i][j][k].letters != other.keyboard[i][j][k].letters:
                        return False
        return True

    def __hash__(self):
        return hash(self.__str__())

    def generate_random_keyboard(self, layout: list[list[int]]):
        letters = get_letters()
        punctuation = get_punctuation()
        random.shuffle(letters)
        random.shuffle(punctuation)

        self.keyboard:list[list[list[Key]]] = [
            [
                [
                    (
                        Key(punctuation.pop())
                        if (i, j, k) in self.punctuation_location
                        else MagicKey(layout) if (i, j, k) in self.magic_locations else Key(letters.pop())
                    )
                    for k in range(column)
                ]
                for j, column in enumerate(hand)
            ]
            for i, hand in enumerate(layout)
        ]

        for hand in self.keyboard:
            for col in hand:
                for key in col:
                    if len(letters) == 0:
                        return
                    elif key.letters[0] not in self.non_letters_set:
                        key.letters+=letters.pop()
                        
    def get_random_letter_kb_index(self) -> tuple[Key, int]:
        while True:
            rand_hand = random.choice(self.keyboard)
            rand_col = random.choice(rand_hand)
            rand_key = random.choice(rand_col)
            if all(char not in self.non_letters_set for char in rand_key.letters):
                break
        if len(rand_key.letters) > 0:
            rand_letter_idx = random.randrange(len(rand_key.letters))
            return (rand_key, rand_letter_idx)
        else:
            raise Exception("Random key doesn't have any letters.")


def test_random_keyboard():
    layout = [[2, 2, 2, 2], [2, 2, 2, 2]]
    kb = Keyboard(layout)
    # Manually check it looks right
    print(str(kb))


# test_random_keyboard()
