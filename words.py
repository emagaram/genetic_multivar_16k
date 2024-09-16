import string
import json
from wordfreq import get_frequency_dict

from settings import USE_PUNCTUATION
from custom_types import FreqDict, FreqList

class CorpusFrequencies():
    def __init__(self) -> None:
        self.inaccuracies_freq = sum(freq for _, freq in create_inaccuracy_freq_list())
        bigrams = get_bigrams()
        trigrams = get_trigrams()
        skipgrams = get_skipgrams()
        characters = get_characters()
        self.bigrams_freq = sum(freq for freq in bigrams.values())
        self.trigrams_freq = sum(freq for freq in trigrams.values())
        self.skipgrams_freq = sum(freq for skipgram_lst in skipgrams for freq in skipgram_lst.values())
        self.letters_freq = sum(freq for skipgram_lst in skipgrams for freq in skipgram_lst.values())
        self.chars_freq = sum(freq for freq in characters.values())

def get_punctuation():
    return list(".,") if USE_PUNCTUATION else list("")
def get_letters():
    return list(string.ascii_lowercase + "'")

def get_letters_and_punctuation():
    return get_letters() + get_punctuation()

def clean_dictionary(word_dict: dict[str, float], letters: list[str]) -> FreqDict:
    return {
        key: value
        for key, value in word_dict.items()
        if all(c in letters for c in key) and key
    }


def create_inaccuracy_freq_dict() -> FreqDict:
    return clean_dictionary(get_frequency_dict(lang="en", wordlist="best"), get_letters())


def create_inaccuracy_freq_list() -> FreqList:
    return sorted(list(create_inaccuracy_freq_dict().items()), key=lambda x: -x[1])

def create_full_freq_dict() -> FreqDict:
    return clean_dictionary(get_frequency_dict(lang="en", wordlist="best"), get_letters() + get_punctuation())


def create_full_freq_list() -> FreqList:
    return sorted(list(create_full_freq_dict().items()), key=lambda x: -x[1])


def get_from_shai(key: str):
    # Open and read the JSON file
    try:
        with open("shai.json", "r") as file:
            data = json.load(file)  # Load the content of the file as a dictionary

        # Extract the bigrams key and return it as a dictionary
        ngrams_dict = data.get(
            key, {}
        )  # Returns an empty dict if 'bigrams' doesn't exist
        ngrams_dict
        return ngrams_dict

    except FileNotFoundError:
        print("The file 'shai.json' was not found.")
        return None
    except json.JSONDecodeError:
        print("Error decoding JSON from 'shai.json'.")
        return None


def get_bigrams() -> dict[str, float]:
    return {
        key: val
        for key, val in get_from_shai("bigrams").items()
        if all(c in get_letters_and_punctuation() for c in key)
    }

def get_trigrams() -> dict[str, float]:
    return {
        key: val
        for key, val in get_from_shai("trigrams").items()
        if all(c in get_letters_and_punctuation() for c in key)
    }


def get_skipgrams() -> list[dict[str, float]]:
    return [
        {
            key: val
            for key, val in get_from_shai(name).items()
            if all(c in get_letters_and_punctuation() for c in key)
        }
        for name in ["skipgrams", "skipgrams2", "skipgrams3"]
    ]

def get_characters() -> dict[str, float]:
    return {
        key: val
        for key, val in get_from_shai("characters").items()
        if all(c in get_letters_and_punctuation() for c in key)
    }