import string
import json
from wordfreq import get_frequency_dict

from constants_weight import USE_PUNCTUATION
from custom_types import FreqDict, FreqList

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


def get_ngrams(key: str):
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
        for key, val in get_ngrams("bigrams").items()
        if all(c in get_letters_and_punctuation() for c in key)
    }

def get_trigrams() -> dict[str, float]:
    return {
        key: val
        for key, val in get_ngrams("trigrams").items()
        if all(c in get_letters_and_punctuation() for c in key)
    }


def get_skipgrams() -> list[dict[str, float]]:
    return [
        {
            key: val
            for key, val in get_ngrams(name).items()
            if all(c in get_letters_and_punctuation() for c in key)
        }
        for name in ["skipgrams", "skipgrams2", "skipgrams3"]
    ]