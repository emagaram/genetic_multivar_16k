import json
import os
import string
from typing import Optional
from custom_types import FreqList
from keyboard import Keyboard
from util import sort_str
from words import create_inaccuracy_freq_list, get_letters_and_punctuation


class InaccuracyEvaluator:
    SHARED_KEY_PENALTIES_FILE = "./shared_key_penalties.json"

    def __init__(self, freq_list: FreqList):
        self.kb: Keyboard = None
        self.shared_key_penalties: dict[str,float] = {}
        self.bigrams = {}
        self.freq_list = freq_list
        self.set_shared_key_penalties()

    def set_kb(self, kb: Keyboard):
        self.kb = kb

    def set_shared_key_penalties(self):
        if os.path.isfile(self.SHARED_KEY_PENALTIES_FILE):
            with open(self.SHARED_KEY_PENALTIES_FILE, "r") as file:
                self.shared_key_penalties = json.load(file)
        else:
            self.calculate_shared_key_penalties()

    def differs(self, w1: str, w2: str) -> Optional[str]:
        res: Optional[str] = None
        for i in range(len(w1)):
            if w1[i] != w2[i]:
                pair = "".join(char for char in sorted([w1[i], w2[i]]))
                if res != None and res != pair:
                    return None
                res = pair
        return res

    def calculate_shared_key_penalties(self):
        words_by_length: dict[int, list[tuple[str, float]]] = {}
        for word, value in self.freq_list:
            w_len = len(word)
            if words_by_length.get(w_len) == None:
                words_by_length[w_len] = []
            words_by_length[w_len].append((word, value))
        total_checks = int(
            sum(
                len(lst) * (len(lst) - 1) / 2 if len(lst) > 0 else 0
                for lst in words_by_length.values()
            )
        )
        count = 0
        for words in words_by_length.values():
            for i, (word_i, freq_i) in enumerate(words):
                for word_j, freq_j in words[i + 1 :]:
                    if count % 100000 == 0:
                        print(
                            f"Progress: {count*100/total_checks:.2f}%, {count}/{total_checks}"
                        )
                    count += 1
                    difference = self.differs(word_i, word_j)
                    if difference != None:
                        if self.shared_key_penalties.get(difference) == None:
                            self.shared_key_penalties[difference] = min(freq_i, freq_j)
                        else:
                            self.shared_key_penalties[difference] += min(freq_i, freq_j)
        with open(self.SHARED_KEY_PENALTIES_FILE, "w") as file:
            json.dump(self.shared_key_penalties, file)

    def evaluate_inaccuracy_heuristic(self, best: float) -> float:
        score = 0
        for hand in self.kb.keyboard:
            for col in hand:
                for key in col:
                    for i, l1 in enumerate(key):
                        for l2 in key[i+1:]:
                            score+=self.shared_key_penalties[sort_str(l1+l2)]
                            if score > best:
                                return score
        # print(f"Heuristic took {count} iterations")
        return score
                    
        

    def evaluate_inaccuracy(
        self,
        best: float,
    ) -> float:
        if self.kb == None:
            raise Exception("Inaccuracy evaluator has no keyboard")
        config = self.kb_to_config()
        score = 0
        index_count = {}

        for i, (word, value) in enumerate(self.freq_list):
            t10_word = self.word_to_t10_word(word, config)
            count = index_count.get(t10_word, 0)
            score += value * min(4, count)
            index_count[t10_word] = count + 1
            if score > best:
                break
        return score

    def kb_to_config(self) -> dict[str, str]:
        keys = list(string.ascii_lowercase + ";")
        char_index = 0
        res = {}
        seen = set()
        for hand in self.kb.keyboard:
            for col in hand:
                for key in col:
                    for char in key:
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

    def get_t10_config_guess_percentage(self, space="") -> str:
        t10_config = self.kb_to_config()
        # Initialize dictionary to store guess counts
        total_word_values = 0
        guess_counts = [0, 0, 0, 0, 0]
        t10_count = {}
        res = ""
        # Evaluate each word
        for word, value in self.freq_list:
            t10_word = self.word_to_t10_word(word, t10_config)
            occurence = t10_count.get(t10_word, 0)
            t10_count[t10_word] = occurence + 1
            total_word_values += value
            if occurence < len(
                guess_counts
            ):  # Check if the word is within the first N guesses
                guess_counts[
                    occurence
                ] += value  # Increment the appropriate guess count

        # Print out the percentages for each guess
        for i, count in enumerate(guess_counts):
            res += f"{space}{['First', 'Second', 'Third', 'Fourth', 'Fifth'][i]} guess percentage: {count*100/total_word_values:.3f}%\n"
        return res


def test_inaccuracy_evaluator():
    inaccuracy_evaluator = InaccuracyEvaluator(create_inaccuracy_freq_list())
    # Case 1: Words differ by only one pair of characters (e.g., "cat" and "bat")
    assert inaccuracy_evaluator.differs("cat", "bat") == "bc", "Test Case 1 Failed"
    # Case 2: Words differ by more than one pair (should return None)
    assert inaccuracy_evaluator.differs("cat", "bog") == None, "Test Case 2 Failed"
    # Case 3: Words are identical (should return None, as no difference)
    assert inaccuracy_evaluator.differs("cat", "cat") == None, "Test Case 3 Failed"
    # Case 4: Words differ in one spot with characters that are already sorted (e.g., "cat" and "cut")
    assert inaccuracy_evaluator.differs("cat", "cut") == "au", "Test Case 4 Failed"
    # Case 5: Words differ in one spot with characters that need to be sorted (e.g., "dog" and "fog")
    assert inaccuracy_evaluator.differs("dog", "fog") == "df", "Test Case 5 Failed"
    # Case 6: Words differ in multiple places but the same pair of characters (should return None)
    assert inaccuracy_evaluator.differs("abab", "baba") == "ab", "Test Case 6 Failed"
    # Case 7: Empty strings (should return None, as no difference can exist)
    assert inaccuracy_evaluator.differs("", "") == None, "Test Case 7 Failed"

    # print("All test cases passed!")


test_inaccuracy_evaluator()
