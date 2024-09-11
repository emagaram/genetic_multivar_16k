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
from discomfort import HindranceEvaluator
from inaccuracy import evaluate_inaccuracy

from finger_freq import FingerFreqEvaluator
from keyboard import RandomKeyboard, Keyboard
from redirect import RedirectEvaluator
from sfb_sfs import SFBSFSEvaluator
from types_1 import FreqList
from words import create_accuracy_freq_list
import constants

zs_sum = 0


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


def write_best_kb_to_file(base_path, score: float, kb: Keyboard, score_file: str):
    # Ensure the Solutions folder exists
    path_name = os.path.join(base_path, "solutions")
    os.makedirs(path_name, exist_ok=True)
    # Path to the file within the Solutions folder
    full_path = os.path.join(path_name, score_file)

    # Write the score and config to the file
    with open(full_path, "w") as file:
        file.write(f"{str(score)}\n{str(kb)}")


def write_best_all_scores_to_file(base_path, score_file: str, all_scores: list[float]):
    # Ensure the Solutions folder exists
    path_name = os.path.join(base_path, "solutions")
    os.makedirs(path_name, exist_ok=True)
    full_path = os.path.join(path_name, score_file)
    # Write the score and config to the file
    with open(full_path, "w") as file:
        json.dump(all_scores, file)


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
            a_col, a_key, a_letter = kb.get_random_non_punc_kb_index()
            b_col, b_key, b_letter = kb.get_random_non_punc_kb_index()
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
            a_col, a_key, a_letter = kb.get_random_non_punc_kb_index()
            b_col, b_key, _ = kb.get_random_non_punc_kb_index()
            while a_col[a_key] == b_col[b_key] or len(a_col[a_key]) == 1:
                a_col, a_key, a_letter = kb.get_random_non_punc_kb_index()
                b_col, b_key, _ = kb.get_random_non_punc_kb_index()
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


def calculate_kb_score(
    kb: Keyboard,
    freq_list: FreqList,
    scores_cache: dict[Keyboard, float],
    finger_freq_evaluator: FingerFreqEvaluator,
    sfb_evaluator: SFBSFSEvaluator,
    discomfort_evaluator: HindranceEvaluator,
    redirect_evaluator: RedirectEvaluator,
    metric: str,
    best: float,
) -> tuple[float, bool]:
    if kb not in scores_cache:
        if metric == "inaccuracy":
            score_tolerance = 1.02
            res = evaluate_inaccuracy(kb, freq_list, score_tolerance * best)
            scores_cache[kb] = res
            if res > score_tolerance * best:
                return (res, False)

        elif metric == "fingerfreq":
            finger_freq_evaluator.set_kb(kb)
            scores_cache[kb] = finger_freq_evaluator.evaluate_finger_frequencies(
                constants.GOAL_FINGER_FREQ
            )
        elif metric == "discomfort":
            discomfort_evaluator.set_kb(kb)
            scores_cache[kb] = sum(
                discomfort_evaluator.evaluate_bigram(bigram)
                for bigram in sfb_evaluator.bigrams.items()
            )
        elif metric == "sfb":
            sfb_evaluator.set_kb(kb)
            scores_cache[kb] = sum(
                sfb_evaluator.evaluate_bigram(bigram)
                for bigram in sfb_evaluator.bigrams.items()
            )
        elif metric == "sfs":
            sfb_evaluator.set_kb(kb)
            scores_cache[kb] = sum(
                sfb_evaluator.evaluate_skipgram((skip_str, skip_freq, i))
                for i, skipgrams in enumerate(sfb_evaluator.skipgrams)
                for skip_str, skip_freq in skipgrams.items()
            )                        
        elif metric == "redirects":
            redirect_evaluator.set_kb(kb)
            res = 0
            score_tolerance = 1.1
            for trigram in redirect_evaluator.trigrams.items():
                res += redirect_evaluator.evaluate_trigram(trigram)
                if res > best * score_tolerance:
                    scores_cache[kb] = res
                    return (res, False)
            scores_cache[kb] = res

        else:
            raise Exception("No case for ", metric)

        return (scores_cache[kb], True)

    return (scores_cache[kb], False)


def run_simulation(
    iteration_path: str, stop_event, layout: list[list[int]], metric: str
):

    POPULATION_SIZE = 60
    SOLUTION_IMPROVEMNT_DEADLINE = 500
    PROCESS_ID = os.getpid()
    ERRORS_LOG_FILENAME = f"{PROCESS_ID}_errors.log"
    all_time_score = sys.float_info.max
    current_best_kb: tuple[float, RandomKeyboard] = (sys.float_info.max, [])
    freq_list = create_accuracy_freq_list()
    finger_freq_evaluator = FingerFreqEvaluator(freq_list)
    sfb_evaluator = SFBSFSEvaluator()
    hind_evaluator = HindranceEvaluator()
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
    all_scores = []
    generation_count = 1
    total_generation_count = 0
    solution_improvement_count = 0
    while not stop_event.is_set():
        if stop_event.is_set():
            break
        if len(all_scores) > 500_000:
            all_scores.sort()
            write_best_all_scores_to_file(iteration_path, f"{metric}.json", all_scores)
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
                f"{metric.capitalize()}: Generation died. Total generations: {generation_count}, Total scores: {len(all_scores)}, Best Score: {current_best_kb[0]}"
            )
            all_scores.sort()
            write_best_all_scores_to_file(iteration_path, f"{metric}.json", all_scores)
            generation_count = 1
            solution_improvement_count = 0
            current_best_kb = (sys.float_info.max, [])
            population = [RandomKeyboard(layout) for _ in range(POPULATION_SIZE)]

        start_time = time.time()  # Start timing the generation
        for kb in population:

            (score, original) = calculate_kb_score(
                kb,
                freq_list,
                scores_cache,
                finger_freq_evaluator,
                sfb_evaluator,
                hind_evaluator,
                redirect_evaluator,
                metric,
                current_best_kb[0],
            )
            if original:
                all_scores.append(score)
            scored_population.append((score, kb))

            if score < current_best_kb[0]:
                current_best_kb = (score, kb)
                solution_improvement_count = 0
                if score < all_time_score:
                    all_time_score = score
                    write_best_kb_to_file(
                        iteration_path,
                        score,
                        kb,
                        metric+"_best_layout.txt",
                    )

        # Sort the population based on the score (lower is better)
        scored_population.sort(key=lambda x: x[0])
        end_time = time.time()
        # print(
        #     f"Generation {generation_count} took {end_time - start_time:.2f} seconds, Best score: {current_best_kb[0]:.8f}"
        # )
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
    # int(cpu_count())
    iteration_id = datetime.datetime.now().strftime("%Y_%m_%d_%Hh_%Mm_%Ss")
    processes = []

    # Create a directory for output if it does not exist
    base_path = "keyboard_optimizer_output_no_full_dict"
    iteration_path = os.path.join(base_path, iteration_id)
    os.makedirs(iteration_path, exist_ok=True)

    layout = [[2, 2, 2, 2], [2, 2, 2, 2]]
    # sample_kb = [["ab", ["cd", "ef"], ["ghi", "j"]]]
    names = ["fingerfreq", "inaccuracy", "sfb", "sfs", "discomfort", "redirects"]
    num_proc = len(names)
    for i in range(num_proc):
        p = Process(
            target=run_simulation,
            args=(iteration_path, stop_event, layout, names[i]),
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
