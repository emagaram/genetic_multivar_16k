import heapq
import json
import os
import sys
from custom_types import FreqList
from keyboard import Key, Keyboard, MagicKey, get_t10_keys
from settings import MODE, InaccuracyMode
import settings
from util import sort_str
from words import CorpusFrequencies, create_inaccuracy_freq_list, get_letters


class InaccuracyEvaluator:

    SHARED_KEY_PENALTIES_FILES: dict[InaccuracyMode, str] = {
        InaccuracyMode.INACCURACY_ALL: f"./shared_key_penalties_all.json",
        InaccuracyMode.INACCURACY_IGNORE_FIRST: "./shared_key_penalties_ignore_first.json",
    }

    def __init__(self, freq_list: FreqList):
        self.kb: Keyboard = None
        self.shared_key_penalties_dicts: dict[InaccuracyMode, dict[str, float]] = {}
        self.bigrams = {}
        self.freq_list = freq_list
        self.corpus_frequencies = CorpusFrequencies()
        self.set_shared_key_penalties()

    def set_kb(self, kb: Keyboard):
        self.kb = kb
        self.kb_config = self.kb_to_config()

    def set_shared_key_penalties(self):
        for mode, file in self.SHARED_KEY_PENALTIES_FILES.items():
            if os.path.isfile(file):
                with open(file, "r") as file:
                    self.shared_key_penalties_dicts[mode] = json.load(file)
            else:
                self.calculate_shared_key_penalties_fast(mode)

    def calculate_shared_key_penalties_fast(
        self, mode: InaccuracyMode
    ) -> dict[str, float]:
        letters = get_letters()
        res: dict[str, float] = {}
        layout = [[[2] + [(len(letters) - 2)]]]
        print(f"Calculating bigram penalties for mode {mode.value}...")
        for i, l1 in enumerate(letters):
            for l2 in letters[i + 1 :]:
                key = sort_str(l1 + l2)
                other_keys = [
                    Key(letter) for letter in letters if letter != l1 and letter != l2
                ]
                kb: list[list[list[Key]]] = [[[Key(key)], other_keys]]
                kb = Keyboard(layout, kb)
                self.set_kb(kb)
                score = self.evaluate_inaccuracy_mode(mode, sys.float_info.max)
                res[key] = score
                print(f"{key}:{score}")
        with open(self.SHARED_KEY_PENALTIES_FILES[mode], "w") as file:
            json.dump(res, file)
        self.shared_key_penalties_dicts[mode] = res

    def evaluate_inaccuracy_heuristic(self, mode: InaccuracyMode, best: float) -> float:
        score = 0
        for hand in self.kb.keyboard:
            for col in hand:
                for key in col:
                    for i, l1 in enumerate(key.letters):
                        for l2 in key.letters[i + 1 :]:
                            score += self.shared_key_penalties_dicts[mode][
                                sort_str(l1 + l2)
                            ]
                            if score > best:
                                return score
        return score

    def evaluate_inaccuracy_mode(self, mode: InaccuracyMode, best: float) -> float:
        if self.kb is None:
            raise Exception("Inaccuracy evaluator has no keyboard")
        score = 0
        index_count = {}
        for word, freq in self.freq_list:
            t10_word = self.word_to_t10_mode(word, self.kb_config, mode)
            count = index_count.get(t10_word, 0)
            score += freq * min(4, count)
            index_count[t10_word] = count + 1
            if score > best:
                break
        return score / self.corpus_frequencies.inaccuracies_freq

    def get_textonyms_heap(self) -> list[tuple[float, list[str], str]]:
        config = self.kb_to_config()
        res = {}
        textonyms: list[tuple[float, list[str], str]] = []
        prev = 0
        for word, freq in self.freq_list:
            t10_word = self.word_to_t10_mode(word, config, MODE)
            lst: list[str] = res.setdefault(t10_word, [])
            lst.append(word)
            if len(lst) > 1:
                heapq.heappush(textonyms, (-freq, lst[:-1], word))
        return textonyms

    def kb_to_config(self) -> dict[str, str]:
        keys = get_t10_keys()
        char_index = 0
        res = {}
        seen = set()
        for hand in self.kb.keyboard:
            for col in hand:
                for key in col:
                    for char in key.letters:
                        res[char] = keys[char_index]
                        seen.add(char)
                    char_index += 1
        return res

    # Function to convert words to T10 digits
    def word_to_t10_word(self, word: str, t10_config: dict[str, str]):
        t10_word = []
        for letter in word:
            t10_word.append(t10_config[letter])
        return "".join(t10_word)

    # Function to convert words to T10 digits
    def word_to_t10_mode(
        self, word: str, t10_config: dict[str, str], mode: InaccuracyMode
    ):
        magic_keys: list[MagicKey] = [
            self.kb.keyboard[hand][col][row]
            for hand, col, row in self.kb.magic_locations
        ]

        def magic_has(current_char: str, prev_t10_key: str):
            for magic_key in magic_keys:
                if magic_key.uses[prev_t10_key] == current_char:
                    return True
            return False

        def append_char(
            lst: list[str],
            t10_config: dict[str, str],
            current_char: str,
            prev_char: str,
        ):
            if magic_has(current_char, t10_config[prev_char]):
                lst.append(current_char)
            else:
                lst.append(t10_config[current_char])

        if mode == InaccuracyMode.INACCURACY_IGNORE_FIRST:
            t10_word = [word[0]]
            for i, letter in enumerate(word[1:], start=1):
                append_char(t10_word, t10_config, word[i], word[i - 1])
        elif mode == InaccuracyMode.INACCURACY_IGNORE_LAST:
            t10_word = []
            for letter in word[:-1]:
                t10_word.append(t10_config[letter])
            t10_word.append(word[-1])
        elif mode == InaccuracyMode.INACCURACY_ONLY_FIRST:
            t10_word = [t10_config[word[0]]]
            for letter in word[1:]:
                t10_word.append(letter)
        elif mode == InaccuracyMode.INACCURACY_MIDDLE:
            t10_word = [word[0]]
            if len(word) > 2:
                for letter in word[1:-1]:
                    t10_word.append(t10_config[letter])
            if len(word) > 1:
                t10_word.append(word[-1])
        elif mode == InaccuracyMode.INACCURACY_ALL:
            t10_word = []
            for letter in word:
                t10_word.append(t10_config[letter])
        return "".join(t10_word)

    def get_t10_config_guess_percentage(self, mode: InaccuracyMode, space="") -> str:
        t10_config = self.kb_to_config()
        # Initialize dictionary to store guess counts
        guess_counts: list[float] = [0, 0, 0, 0, 0]
        index_count = {}
        res = ""
        # Evaluate each word
        for word, freq in self.freq_list:
            t10_word = self.word_to_t10_mode(word, t10_config, mode)
            count = index_count.get(t10_word, 0)
            if count < len(
                guess_counts
            ):  # Check if the word is within the first N guesses
                guess_counts[count] += (
                    freq / self.corpus_frequencies.inaccuracies_freq
                )  # Increment the appropriate guess count
            index_count[t10_word] = count + 1
        # Print out the percentages for each guess
        for i, count in enumerate(guess_counts):
            res += f"{space}{['First', 'Second', 'Third', 'Fourth', 'Fifth'][i]} guess percentage: {count*100:.7f}%\n"
        return res

    def sample_n_words(self, n: int, mode: InaccuracyMode, space=" "):
        res = ""
        for word, _ in self.freq_list[:n]:
            t10_word = self.word_to_t10_mode(word, self.kb_config, mode)
            res += f"{space}{word}:{t10_word}\n"
        return res


def test_inaccuracy_evaluator():
    print("All inaccuracy test cases passed!")


# test_inaccuracy_evaluator()


# def artificial(self):
#     return {
#         "m": "a",
#         "x": "a",
#         "g": "a",
#         "f": "b",
#         "d": "b",
#         "w": "c",
#         "a": "c",
#         "k": "c",
#         "b": "d",
#         "n": "d",
#         "q": "d",
#         "c": "e",
#         "e": "e",
#         "s": "f",
#         "u": "f",
#         "v": "g",
#         "h": "g",
#         "i": "g",
#         "z": "h",
#         "r": "h",
#         "p": "h",
#         "j": "i",
#         "l": "i",
#         "y": "i",
#         "t": "j",
#         "o": "j",
#         "'": "j",
#     }
