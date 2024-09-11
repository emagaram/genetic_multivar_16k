import string
from keyboard import RandomKeyboard

def evaluate_inaccuracy(
    kb: RandomKeyboard,
    w_freq_list: list[tuple[str, float]],
    best: float,
) -> float:
    config = kb_to_config(kb)
    score = 0
    index_count = {}
    for word, value in w_freq_list:
        t10_word = word_to_t10_word(word, config)
        count = index_count.get(t10_word, 0)
        score += value * min(4, count)
        index_count[t10_word] = count + 1
        if score > best:
            break
    return score


def kb_to_config(kb:RandomKeyboard)->dict[str,str]:
    KB_KEYS = list(string.ascii_lowercase + ";")
    char_index = 0
    res = {}
    seen = set()
    for hand in kb.keyboard:
        for col in hand:
            for key in col:
                for char in key:
                    res[char] = KB_KEYS[char_index]
                    seen.add(char)
                    # print("Key",key,"Setting char:",char,"to key",KB_KEYS[char_index])
                char_index+=1
    # print(len(seen))
    return res
        
    

# Function to convert words to T10 digits
def word_to_t10_word(word: str, t10_config: dict[str, str]):
    t10_word = []
    for letter in word:
        t10_word.append(t10_config[letter])
    return "".join(t10_word)
