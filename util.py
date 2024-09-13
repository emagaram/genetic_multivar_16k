import math
from keyboard import Keyboard


def sort_str(s: str):
    return "".join(sorted(s))


def kb_to_column_dict(kb: Keyboard) -> dict[str, int]:
    res = {}
    index = 0
    for hand in kb.keyboard:
        for col in hand:
            for key in col:
                for char in key:
                    res[char] = index
            index += 1
    return res


def kb_to_reverse_column_dict(kb) -> dict[int, str]:
    res:dict[int, str] = {}
    index = 0
    for hand in kb.keyboard:
        for col in hand:
            for key in col:
                for char in key:
                    if res.get(index) == None:
                        res[index] = char
                    else:
                        res[index] += char
            index += 1
    return res


def kb_to_row_dict(kb: Keyboard) -> dict[str, int]:
    res = {}
    for hand in kb.keyboard:
        for col in hand:
            for i, key in enumerate(col):
                for char in key:
                    res[char] = i
    return res


def calculate_mean(data: list):
    return sum(data) / len(data)


def calculate_std(data: list, mean: float):
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return math.sqrt(variance)


def calculate_zscore(x: float, mean: float, std: float):
    return (x - mean) / std
