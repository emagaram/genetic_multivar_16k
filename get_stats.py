import heapq
from discomfort import DiscomfortEvaluator
from finger_freq import FingerFreqEvaluator
from inaccuracy import InaccuracyEvaluator
from keyboard import Keyboard
from sfb_sfs import SFBSFSEvaluator
from rolls_alts_redirects import RAREvaluator
from words import create_inaccuracy_freq_list
from settings import GOAL_FINGER_FREQ, GOAL_FINGER_MAX, MODE, InaccuracyMode


def get_score_stats(kb: Keyboard, performance: dict[str, float]):
    space = "  "
    res = f"Inaccuracy Mode: {MODE.value}\nRaw Scores:\n"
    
    
    for stat_name, raw_score in performance.items():
        res+=f"{space}{stat_name.capitalize()}: {raw_score}\n"
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
            res+=f"{space}{side} {finger}: {freq*100:.3f}%, Max: {GOAL_FINGER_MAX[i][j]*100:.3f}%\n"
    res+="\n"
    
    # SFB
    sfb_evaluator = SFBSFSEvaluator(kb)
    sfb_sum_only1u = sum(sfb_evaluator.evaluate_bigram_stat(bigram, True, sfb_evaluator.corpus_frequencies.bigrams_freq) for bigram in sfb_evaluator.bigrams.items())
    res+=f"1U Same Finger Bigrams: {sfb_sum_only1u*100:.3f}%\n\n"
    sfb_sum = sum(sfb_evaluator.evaluate_bigram_stat(bigram, False, sfb_evaluator.corpus_frequencies.bigrams_freq) for bigram in sfb_evaluator.bigrams.items())
    res+=f"0U and 1U Same Finger Bigrams: {sfb_sum*100:.3f}%\n\n"
    
    # SFS 
    for i, skipgrams in enumerate(sfb_evaluator.skipgrams):
        sfs_sum = sum(sfb_evaluator.evaluate_skipgram_stat(skipgram, True, i) for skipgram in skipgrams.items())
        res+=f"1U Same Finger Skipgrams-{i}: {sfs_sum*100:.3f}%\n"
    res+="\n"
    
    for i, skipgrams in enumerate(sfb_evaluator.skipgrams):
        sfs_sum = sum(sfb_evaluator.evaluate_skipgram_stat(skipgram, False, i) for skipgram in skipgrams.items())
        res+=f"0U and 1U Same Finger Skipgrams-{i}: {sfs_sum*100:.3f}%\n"
    res+="\n"    
    
    # Discomfort
    discomfort_evaluator = DiscomfortEvaluator(kb)
    discomfort_sum = sum(discomfort_evaluator.evaluate_bigram_stat(bigram) for bigram in sfb_evaluator.bigrams.items())
    res+=f"Discomfort: {discomfort_sum*100:.3f}%\n\n"
    
    # Rolls, Alts, Redirects
    rar_evaluator = RAREvaluator(kb)
    rolls, alts, redirects = rar_evaluator.evaluate_rolls_alts_redirects_stat()    
    res+=f"Rolls: {rolls*100:.3f}%\n"
    res+=f"Alts: {alts*100:.3f}%\n"
    res+=f"Redirects: {redirects*100:.3f}%\n\n"
        
    # Inaccuracy
    inaccuracy_evaluator = InaccuracyEvaluator(create_inaccuracy_freq_list())
    inaccuracy_evaluator.set_kb(kb)
    res+=f"Inaccuracy {MODE.value}:\n{inaccuracy_evaluator.get_t10_config_guess_percentage(MODE, space)}\n"
    if MODE != InaccuracyMode.INACCURACY_ALL:
        res+=f"Inaccuracy {InaccuracyMode.INACCURACY_ALL.value}:\n{inaccuracy_evaluator.get_t10_config_guess_percentage(InaccuracyMode.INACCURACY_ALL, space)}\n"

    
    # Top textonyms
    res+=f"Top Texonyms Max:\n"
    top_textonyms = inaccuracy_evaluator.get_textonyms_heap()
    freq_sum = 0
    top_n = 100
    for _ in range(top_n):
        (freq, words, word) = heapq.heappop(top_textonyms)
        freq*=-1 # To make positive
        res+=f"{space}{word}({freq}):{words}\n"
        freq_sum += freq
    res+=f"Top {top_n} total frequency: {100*freq_sum}%\n"
    
    
    return res

