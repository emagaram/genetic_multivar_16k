import json
import logging
import math
import random
import time
import sys
import os
import copy
from enum import Enum

import datetime
from threading import Thread
from multiprocessing import Event, Process, cpu_count
from discomfort import DiscomfortEvaluator
from inaccuracy import evaluate_inaccuracy

from util import calculate_mean, calculate_std, calculate_zscore
from finger_freq import FingerFreqEvaluator
from keyboard import RandomKeyboard, Keyboard
from redirect import RedirectEvaluator
from sfb_sfs import SFBSFSEvaluator
from custom_types import FreqList
from words import create_inaccuracy_freq_list
import constants_rank


def monitor_for_termination(stop_event):
    """Monitor for the 'end' command in the console to terminate the processes."""
    while not stop_event.is_set():
        terminate_command = input()
        if terminate_command.lower() == "end":
            stop_event.set()


def write_end_time(base_path: str):
    """Write the current end time to a file."""
    end_time = datetime.datetime.now().strftime("%Y_%m_%d_%Hh_%Mm_%Ss")
    filename = os.path.join(base_path, "end_time.txt")
    with open(filename, "w") as file:
        file.write(end_time)


def read_best_score_from_file(base_path, score_file: str):
    # Initialize default values
    best_score = sys.float_info.max
    best_config = None
    path_name = os.path.join(base_path, "solutions", score_file)
    # Check if the file exists
    if os.path.exists(path_name):
        with open(path_name, "r") as file:
            data = file.readlines()

            # Update scores if file is properly formatted
            if len(data) >= 2:  # Expecting two scores and one config
                best_score = float(data[0].strip())
                best_config = eval(data[1].strip())  # Convert string back to dictionary

    return best_score, best_config


def write_best_kb_to_file(
    base_path, performance: dict[str, float], kb: Keyboard, score_file: str
):
    # Ensure the Solutions folder exists
    path_name = os.path.join(base_path, "solutions")
    os.makedirs(path_name, exist_ok=True)
    # Path to the file within the Solutions folder
    full_path = os.path.join(path_name, score_file)

    # Write the score and config to the file
    with open(full_path, "w") as file:
        file.write(str(kb) + "\n")
        for key, val in performance.items():
            file.write(f"{key}: {val}\n")


# [[2,2,2,2],[2,2,2,2]]
punctuation_options_2_symbols: list[
    tuple[tuple[int, int, int], tuple[int, int, int]]
] = [
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
    ((0, 0, 0), (1, 3, 0)),
]


def get_similarity_dict(kbs: list[RandomKeyboard]) -> dict[frozenset[str], int]:
    kb_diversity = {}
    for kb in kbs:
        kb_diversity[kb] = kb_diversity.get(kb, 0) + 1
    # Return the number of unique configurations
    return kb_diversity


# Moves a letter to another key
def mutate(kb: RandomKeyboard):
    for _ in range(random.randint(1, 5)):

        # Swap
        if random.random() < 0.5:
            a_col, a_key, a_letter = kb.get_random_letter_kb_index()
            b_col, b_key, b_letter = kb.get_random_letter_kb_index()
            # while a_letter == -1:
            #     a_col, a_key, a_letter = get_random_kb_index(kb)
            # while b_letter == -1:
            #     b_col, b_key, b_letter = get_random_kb_index(kb)
            a_char = a_col[a_key][a_letter]
            b_char = b_col[b_key][b_letter]
            a_col[a_key] = a_col[a_key].replace(a_char, b_char, 1)
            b_col[b_key] = b_col[b_key].replace(b_char, a_char, 1)
        # Move
        else:
            a_col, a_key, a_letter = kb.get_random_letter_kb_index()
            b_col, b_key, _ = kb.get_random_letter_kb_index()
            while a_col[a_key] == b_col[b_key] or len(a_col[a_key]) == 1:
                a_col, a_key, a_letter = kb.get_random_letter_kb_index()
                b_col, b_key, _ = kb.get_random_letter_kb_index()
            a_char = a_col[a_key][a_letter]
            a_col[a_key] = a_col[a_key].replace(a_char, "", 1)
            b_col[b_key] += a_char


def get_index_from_percent(percent, total_freq, freq_list):
    reg_total_freq = percent * total_freq
    sum = 0
    reg_list_index = -1
    for index, (_, freq) in enumerate(freq_list):
        sum += freq
        if sum >= reg_total_freq:
            reg_list_index = index
            break
    if reg_list_index == -1:
        print(f"Miscalculating index. Percent: {percent}")
    return reg_list_index


def write_generations_completed(base_path: str, process_id: str, generations: int):
    directory_path = os.path.join(base_path, "performance")
    os.makedirs(directory_path, exist_ok=True)
    filename = os.path.join(directory_path, f"{process_id}_generations_completed.txt")
    with open(filename, "w") as file:
        file.write(f"{generations}")


def write_to_file(base_path: str, file_name: str, text: str):
    directory_path = os.path.join(base_path)
    os.makedirs(directory_path, exist_ok=True)
    filename = os.path.join(directory_path, file_name)
    with open(filename, "w") as file:
        file.write(text)

def normalize_score(score: float, min_score: float):
    return math.sqrt(score)/math.sqrt(min_score)

def calculate_min_failing_score(best: float, current: float, min: float, mult: float):
    add = best - current
    add /= mult
    return (add*math.sqrt(min))**2

def calculate_kb_score(
    kb: Keyboard,
    freq_list: FreqList,
    best: float,
    mins: list,
    scores_cache: dict[Keyboard, dict[str, float]],
    finger_freq_evaluator: FingerFreqEvaluator,
    sfb_evaluator: SFBSFSEvaluator,
    hind_evaluator: DiscomfortEvaluator,
    redirect_evaluator: RedirectEvaluator,
) -> dict[str, float]:

    # Space key 18.5%
    score = 0
    if kb not in scores_cache:
        finger_freq_evaluator.set_kb(kb)
        sfb_evaluator.set_kb(kb)
        # hind_evaluator.set_kb(kb)
        # redirect_evaluator.set_kb(kb)
        acc = evaluate_inaccuracy(kb, freq_list, best) if mins["accuracy"][1] else 0
        score += constants_rank.INACCURACY_WEIGHT * acc

        sfb_sum = 0
        hind_sum = 0
        redirect_sum = 0
        if mins["redirects"][1]:
            for trigram in redirect_evaluator.trigrams.items():
                redirect_sum += (
                    redirect_evaluator.evaluate_trigram(trigram)
                    if mins["redirects"][1]
                    else 0
                )

        score += constants_rank.REDIRECT_WEIGHT * redirect_sum
        if mins["sfbs"][1] or mins["hinds"][1]:
            for bigram in sfb_evaluator.bigrams.items():
                sfb_sum += (
                    sfb_evaluator.evaluate_bigram(bigram) if mins["sfbs"][1] else 0
                )
                hind_sum += (
                    hind_evaluator.evaluate_bigram(bigram) if mins["hinds"][1] else 0
                )
            for i, skipgrams in enumerate(sfb_evaluator.skipgrams):
                for skip_str, skip_freq in skipgrams.items():
                    sfb_sum += (
                        sfb_evaluator.evaluate_skipgram((skip_str, skip_freq, i))
                        if mins["sfbs"][1]
                        else 0
                    )

        score += constants_rank.DISCOMFORT_WEIGHT * hind_sum
        score += constants_rank.SFB_WEIGHT * sfb_sum
        ff_sum = (
            finger_freq_evaluator.evaluate_finger_frequencies_MSE(
                constants_rank.GOAL_FINGER_FREQ
            )
            if mins["ffs"][1]
            else 0
        )

        score += constants_rank.FINGER_FREQ_WEIGHT * ff_sum
        scores_cache[kb] = {
            "score": score,
            "ffs": ff_sum,
            "acc": acc,
            "hin": hind_sum,
            "sfb": sfb_sum,
            "red": redirect_sum,
        }
        if mins["accuracy"][1]:
            mins["accuracy"][0] = min(acc, mins["accuracy"][0])
        if mins["redirects"][1]:
            mins["redirects"][0] = min(redirect_sum, mins["redirects"][0])
        if mins["sfbs"][1]:
            mins["sfbs"][0] = min(sfb_sum, mins["sfbs"][0])
        if mins["hinds"][1]:
            mins["hinds"][0] = min(hind_sum, mins["hinds"][0])
        if mins["ffs"][1]:
            mins["ffs"][0] = min(ff_sum, mins["ffs"][0])

    return scores_cache[kb]


# Function to calculate harmonic mean
# # Example usage
# values = [2, 5, 10, 0]  # The dataset with a zero value
# normalized_values = ratio_normalization(values)

# # Output the normalized values
# print("Original values:", values)
# print("Normalized values:", normalized_values)


def run_simulation(
    iteration_path: str,
    stop_event,
    layout: list[list[int]],
):

    mins: dict[list[float, bool]] = {}

    for name in ["accuracy", "ffs", "hinds", "redirects", "sfbs"]:
        mins[name] = [sys.float_info.max, False]
    mins["sfbs"][1] = True

    NUM_KEYS = sum(i for row in layout for i in row)
    POPULATION_SIZE = 60
    SOLUTION_IMPROVEMNT_DEADLINE = 500
    PROCESS_ID = os.getpid()
    ERRORS_LOG_FILENAME = f"{PROCESS_ID}_errors.log"
    SCORE_FILE = f"{PROCESS_ID}_best_{NUM_KEYS}_key_layout.txt"
    all_time_score = sys.float_info.max
    current_best_kb: tuple[float, RandomKeyboard] = (sys.float_info.max, [])
    freq_list = create_inaccuracy_freq_list()
    finger_freq_evaluator = FingerFreqEvaluator(freq_list)
    sfb_evaluator = SFBSFSEvaluator()
    hind_evaluator = DiscomfortEvaluator()
    redirect_evaluator = RedirectEvaluator()
    scores_cache: dict[Keyboard, dict[str, float]] = {}
    errors_path = os.path.join(iteration_path, "errors")
    os.makedirs(errors_path, exist_ok=True)
    log_file_path = os.path.join(errors_path, ERRORS_LOG_FILENAME)
    logging.basicConfig(
        filename=log_file_path,
        level=logging.ERROR,
        format="%(asctime)s:%(levelname)s:%(message)s",
    )
    population: list[RandomKeyboard] = [
        RandomKeyboard(layout) for _ in range(POPULATION_SIZE)
    ]
    generation_count = 1
    total_generation_count = 0
    solution_improvement_count = 0
    # start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Readable start time
    while not stop_event.is_set():
        if stop_event.is_set():
            break
        scored_population: list[tuple[float, RandomKeyboard]] = []

        # Called when no new improvements have been made in a while
        if solution_improvement_count > SOLUTION_IMPROVEMNT_DEADLINE:
            scores_cache.clear()
            total_generation_count += generation_count
            write_generations_completed(
                iteration_path, PROCESS_ID, total_generation_count
            )
            print(
                f"Generation died. Total generations: {generation_count}, Reg Score: {current_best_kb[0]}"
            )
            generation_count = 1
            solution_improvement_count = 0
            current_best_kb = (sys.float_info.max, [])
            population = [RandomKeyboard(layout) for _ in range(POPULATION_SIZE)]

        start_time = time.time()  # Start timing the generation
        for kb in population:

            performance = calculate_kb_score(
                kb,
                freq_list,
                current_best_kb[0],
                mins,
                scores_cache,
                finger_freq_evaluator,
                sfb_evaluator,
                hind_evaluator,
                redirect_evaluator,
            )
            score = performance["score"]
            scored_population.append((score, kb))

            if score < current_best_kb[0]:
                current_best_kb = (score, kb)
                solution_improvement_count = 0
                if score < all_time_score:
                    all_time_score = score
                    write_best_kb_to_file(
                        iteration_path,
                        performance,
                        kb,
                        SCORE_FILE,
                    )

        # Sort the population based on the score (lower is better)
        scored_population.sort(key=lambda x: x[0])
        end_time = time.time()
        print(
            f"Generation {generation_count} took {end_time - start_time:.2f} seconds, Best score: {current_best_kb[0]:.8f}"
        )
        generation_count += 1
        solution_improvement_count += 1

        # Selection: Top 1/4 automatically admitted, then new are generated from all
        next_population = [
            config for _, config in scored_population[: len(scored_population) // 4]
        ]
        # seen = set(accuracy_scores_cache.keys())
        # TODO, copy needed here? remove?
        potential_parents = [config for _, config in scored_population]
        random.shuffle(potential_parents)

        while len(next_population) < POPULATION_SIZE:
            # Shuffle the potential parent pool every time to ensure randomness
            parent = potential_parents.pop()
            child = copy.deepcopy(parent)
            mutate(child)
            next_population.append(child)
        population = next_population


if __name__ == "__main__":
    # Setup stop event

    stop_event = Event()

    # Start the termination monitor in a separate thread
    termination_thread = Thread(target=monitor_for_termination, args=(stop_event,))
    termination_thread.start()

    # Modify your main script logic here
    num_processes = 1  # int(cpu_count())
    iteration_id = datetime.datetime.now().strftime("%Y_%m_%d_%Hh_%Mm_%Ss")
    processes = []

    # Create a directory for output if it does not exist
    base_path = "keyboard_optimizer_output_no_full_dict"
    iteration_path = os.path.join(base_path, iteration_id)
    os.makedirs(iteration_path, exist_ok=True)

    layout = [[2, 2, 2, 2], [2, 2, 2, 2]]
    # sample_kb = [["ab", ["cd", "ef"], ["ghi", "j"]]]

    for i in range(num_processes):
        p = Process(
            target=run_simulation,
            args=(
                iteration_path,
                stop_event,
                layout,
            ),
        )
        p.start()
        processes.append(p)
        print(f"Started optimizer {i}")

    try:
        for p in processes:
            p.join()  # Wait for all processes to complete
    except KeyboardInterrupt:
        write_end_time(iteration_path)
    finally:
        # Ensure the input thread is also terminated
        termination_thread.join()
        # Finally, write the end time regardless of how the program exits
        write_end_time(iteration_path)
        print(f"End time written to {os.path.join(iteration_path, 'end_time.txt')}")

def test_min_failing_score():
    best = 2.7
    current = 1.5
    metric_min = 0.2
    mult = 0.5
    mfs = calculate_min_failing_score(best, current, metric_min, mult)
    add = normalize_score(mfs, metric_min)
    add*=mult
    assert(add+current <= best)
    assert(add+current+0.0001 > best)


test_min_failing_score()