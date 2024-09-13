from finger_freq import FingerFreqEvaluator
from keyboard import Keyboard
from words import create_inaccuracy_freq_list


def get_score_stats(kb: Keyboard, performance: dict[str, tuple[float, float]]):
    res = ""
    space = "  "
    for stat_name, (raw_score, rank) in performance:
        res+=f"{stat_name}:\n"
        res+=f"{space}raw score: {raw_score}\n"
        if stat_name!="score":
            res+=f"{space}rank: {rank}\n"
    res+="\n"
    
    # Finger usage
    fingerfreq_evaluator = FingerFreqEvaluator(create_inaccuracy_freq_list())
    fingerfreq_evaluator.set_kb(kb)
    finger_freqs = fingerfreq_evaluator.get_finger_frequencies()
    res+="Finger Usage:\n"
    fingers = ["Pinky", "Ring", "Middle", "Index"]
    fingers_rev = fingers[::-1]
    for i, hand in enumerate(finger_freqs):
        for j, freq in enumerate(hand):
            side = "Left" if i == 0 else "Right"
            finger = fingers[j] if i == 0 else fingers_rev[j]
            res+=f"{space}{side} {finger}: {freq}\n"
    res+="\n"
    
    # SFB
    
     
    return res
