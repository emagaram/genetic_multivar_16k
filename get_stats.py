from discomfort import DiscomfortEvaluator
from finger_freq import FingerFreqEvaluator
from inaccuracy import InaccuracyEvaluator
from keyboard import Keyboard
from redirect import RedirectEvaluator
from sfb_sfs import SFBSFSEvaluator
from words import create_inaccuracy_freq_list
from settings import GOAL_FINGER_FREQ

def get_score_stats(kb: Keyboard, performance: dict[str, float]):
    space = "  "
    res = ""
    for stat_name, raw_score in performance.items():
        res+=f"{stat_name.capitalize()}:\n"
        res+=f"{space}raw score: {raw_score}\n"
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
            res+=f"{space}{side} {finger}: {freq*100:.3f}%, Goal: {GOAL_FINGER_FREQ[i][j]*100:.3f}%\n"
    res+="\n"
    
    # SFB
    sfb_evaluator = SFBSFSEvaluator(kb)
    sfb_sum_only1u = sum(sfb_evaluator.evaluate_bigram_stat(bigram, True) for bigram in sfb_evaluator.bigrams.items())
    res+=f"1U Same Finger Bigrams: {sfb_sum_only1u*100:.3f}%\n\n"
    sfb_sum_only1u = sum(sfb_evaluator.evaluate_bigram_stat(bigram, False) for bigram in sfb_evaluator.bigrams.items())
    res+=f"0U and 1U Same Finger Bigrams: {sfb_sum_only1u*100:.3f}%\n\n"
    
    # SFS 
    for i, skipgrams in enumerate(sfb_evaluator.skipgrams):
        sfs_sum = sum(sfb_evaluator.evaluate_skipgram_stat(skipgram, True) for skipgram in skipgrams.items())
        res+=f"1U Same Finger Skipgrams-{i}: {sfs_sum*100:.3f}%\n"
    res+="\n"
    
    for i, skipgrams in enumerate(sfb_evaluator.skipgrams):
        sfs_sum = sum(sfb_evaluator.evaluate_skipgram_stat(skipgram, False) for skipgram in skipgrams.items())
        res+=f"0U and 1U Same Finger Skipgrams-{i}: {sfs_sum*100:.3f}%\n"
    res+="\n"    
    
    # Discomfort
    discomfort_evaluator = DiscomfortEvaluator(kb)
    discomfort_sum = sum(discomfort_evaluator.evaluate_bigram_stat(bigram) for bigram in sfb_evaluator.bigrams.items())
    res+=f"Discomfort: {discomfort_sum*100:.3f}%\n\n"
    
    # Inaccuracy
    inaccuracy_evaluator = InaccuracyEvaluator(create_inaccuracy_freq_list())
    inaccuracy_evaluator.set_kb(kb)
    res+=f"Inaccuracy:\n{inaccuracy_evaluator.get_t10_config_guess_percentage(space)}\n"
    
    # Redirects
    redirect_evaluator = RedirectEvaluator(kb)
    redirect_sum = sum(redirect_evaluator.evaluate_trigram_stat(trigram) for trigram in redirect_evaluator.trigrams.items())
    res+=f"Redirects: {redirect_sum*100:.3f}%\n\n"
    return res


