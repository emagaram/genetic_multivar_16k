import json
import logging
import bisect
import math
import random
import time
import sys
import os
import copy


import datetime
from threading import Thread
from multiprocessing import Event, Process, cpu_count
from discomfort import DiscomfortEvaluator
from inaccuracy import InaccuracyEvaluator, evaluate_inaccuracy
from score_categories import Categories
import archive.constants_rank as constants_rank

from finger_freq import FingerFreqEvaluator
from keyboard import RandomKeyboard, Keyboard
from redirect import RedirectEvaluator
from sfb_sfs import SFBSFSEvaluator
from custom_types import FreqList
from words import create_full_freq_list, create_inaccuracy_freq_list

def shorten_pid(pid: int) -> str:
    return str(pid % 100)

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
    base_path,
    performance: dict[str, tuple[float, float]],
    kb: Keyboard,
    score_file: str,
):
    # Ensure the Solutions folder exists
    path_name = os.path.join(base_path, "solutions")
    os.makedirs(path_name, exist_ok=True)
    # Path to the file within the Solutions folder
    full_path = os.path.join(path_name, score_file)
    space = "  "
    # Write the score and config to the file
    with open(full_path, "w") as file:
        file.write(str(kb) + "\n")
        for key, (val, rank) in performance.items():
            file.write(f"{key}:\n{space}raw score:{val}\n{space}rank:{rank}\n\n")

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

# Moves a letter to another key
def mutate(kb: RandomKeyboard):
    for _ in range(random.randint(1, 5)):

        # Swap
        if random.random() < 0.5:
            a_col, a_key, a_letter = kb.get_random_letter_kb_index()
            b_col, b_key, b_letter = kb.get_random_letter_kb_index()
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


def get_rank(score: float, score_list: list[float]) -> float:
    return bisect.bisect_left(score_list, score) / len(score_list)


def get_rank_inverse(target_rank: float, score_list: list[float]) -> float:
    index = max(math.ceil(len(score_list) * target_rank), 0)
    return score_list[min(index, len(score_list) - 1)]


def calculate_kb_score(
    kb: Keyboard,
    best: float,
    scores_cache: dict[Keyboard, dict[str, tuple[float, float]]],
    finger_freq_evaluator: FingerFreqEvaluator,
    sfb_evaluator: SFBSFSEvaluator,
    discomfort_evaluator: DiscomfortEvaluator,
    redirect_evaluator: RedirectEvaluator,
    inaccuracy_evaluator: InaccuracyEvaluator,
    scores_dict: dict[Categories, list[float]],
) -> dict[str, tuple[float, float]]:

    score = 0
    if kb not in scores_cache:
        finger_freq_evaluator.set_kb(kb)
        sfb_evaluator.set_kb(kb)
        discomfort_evaluator.set_kb(kb)
        redirect_evaluator.set_kb(kb)
        inaccuracy_evaluator.set_kb(kb)
        score = 0
        redirect_sum = 0
        for trigram in redirect_evaluator.trigrams.items():
            redirect_sum += redirect_evaluator.evaluate_trigram(trigram)
            if (
                score
                + constants_rank.REDIRECT_WEIGHT
                * get_rank(redirect_sum, scores_dict[Categories.REDIRECT])
                > best
            ):
                break
        redirect_rank = get_rank(redirect_sum, scores_dict[Categories.REDIRECT])
        score += constants_rank.REDIRECT_WEIGHT * redirect_rank
        sfb_sum = 0
        discomfort_sum = 0
        for bigram in sfb_evaluator.bigrams.items():
            sfb_sum += sfb_evaluator.evaluate_bigram(bigram)
            discomfort_sum += discomfort_evaluator.evaluate_bigram(bigram)
            if (
                score
                + constants_rank.SFB_WEIGHT * get_rank(sfb_sum, scores_dict[Categories.SFB])
                + constants_rank.DISCOMFORT_WEIGHT
                * get_rank(discomfort_sum, scores_dict[Categories.DISCOMFORT])
                > best
            ):
                break
        discomfort_rank = get_rank(discomfort_sum, scores_dict[Categories.DISCOMFORT])
        sfb_rank = get_rank(sfb_sum, scores_dict[Categories.SFB])
        score += constants_rank.DISCOMFORT_WEIGHT * discomfort_rank
        score += constants_rank.SFB_WEIGHT * sfb_rank
        sfs_sum = 0
        for i, skipgrams in enumerate(sfb_evaluator.skipgrams):
            for skip_str, skip_freq in skipgrams.items():
                sfs_sum += sfb_evaluator.evaluate_skipgram((skip_str, skip_freq, i))
                if (
                    score
                    + constants_rank.SFS_WEIGHT
                    * get_rank(sfs_sum, scores_dict[Categories.SFS])
                    > best
                ):
                    break
        sfs_rank = get_rank(sfs_sum, scores_dict[Categories.SFS])
        score += constants_rank.SFS_WEIGHT * sfs_rank

        fingerfreq_sum = finger_freq_evaluator.evaluate_finger_frequencies_MSE(
            constants_rank.GOAL_FINGER_FREQ
        )
        fingerfreq_rank = get_rank(fingerfreq_sum, scores_dict[Categories.FINGERFREQ])
        score += constants_rank.FINGER_FREQ_WEIGHT * fingerfreq_rank
        # 0.8 - 0.6 = 0.2
        # x/const.inacc_weight = 0.2
        minimum_failing_inaccuracy_score = (
            get_rank_inverse(
                (best - score) / constants_rank.INACCURACY_WEIGHT,
                scores_dict[Categories.INACCURACY],
            )
        ) + 0.0001
        # print("MFI:", minimum_failing_inaccuracy_score)
        # print("MFI Rank:", get_rank(
        #     minimum_failing_inaccuracy_score, scores_dict[Categories.INACCURACY]
        # ))
        testsum = score + constants_rank.INACCURACY_WEIGHT * get_rank(
            minimum_failing_inaccuracy_score, scores_dict[Categories.INACCURACY]
        )
        # print("MFI Sum", testsum)
        # print("Best:", best)
        # print("MFI Sum > Best:", testsum>best)
        inaccuracy_sum = inaccuracy_evaluator.evaluate_inaccuracy(best)
        inaccuracy_rank = get_rank(inaccuracy_sum, scores_dict[Categories.INACCURACY])
        score += constants_rank.INACCURACY_WEIGHT * inaccuracy_rank

        scores_cache[kb] = {
            "score": (score, score),
            Categories.FINGERFREQ.value: (fingerfreq_sum, fingerfreq_rank),
            Categories.INACCURACY.value: (inaccuracy_sum, inaccuracy_rank),
            Categories.DISCOMFORT.value: (discomfort_sum, discomfort_rank),
            Categories.SFB.value: (sfb_sum, sfb_rank),
            Categories.SFS.value: (sfs_sum, sfs_rank),
            Categories.REDIRECT.value: (redirect_sum, redirect_rank),
        }

    return scores_cache[kb]


def run_simulation(
    iteration_path: str,
    stop_event,
    layout: list[list[int]],
):
    scores_dict: dict[Categories, list[float]] = {}

    for name in Categories:
        with open(f"./starting_data/{name.value}.json", "r") as file:
            scores_dict[name] = json.load(file)

    POPULATION_SIZE = 60
    SOLUTION_IMPROVEMNT_DEADLINE = 500
    PROCESS_ID = shorten_pid(os.getpid())
    ERRORS_LOG_FILENAME = f"{PROCESS_ID}_errors.log"
    SCORE_FILE = f"{PROCESS_ID}_best_layout.txt"
    all_time_score = sys.float_info.max
    current_best_kb: tuple[float, RandomKeyboard] = (10000000, [])
    finger_freq_evaluator = FingerFreqEvaluator(create_full_freq_list())
    sfb_evaluator = SFBSFSEvaluator()
    discomfort_evaluator = DiscomfortEvaluator()
    redirect_evaluator = RedirectEvaluator()
    inaccuracy_evaluator = InaccuracyEvaluator(create_inaccuracy_freq_list())
    scores_cache: dict[Keyboard, dict[str, tuple[float, float]]] = {}
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
            current_best_kb = (10000, [])
            population = [RandomKeyboard(layout) for _ in range(POPULATION_SIZE)]

        start_time = time.time()  # Start timing the generation
        for kb in population:
            performance = calculate_kb_score(
                kb,
                current_best_kb[0],
                scores_cache,
                finger_freq_evaluator,
                sfb_evaluator,
                discomfort_evaluator,
                redirect_evaluator,
                inaccuracy_evaluator,
                scores_dict
            )
            score = performance["score"][0]
            scored_population.append((score, kb))

            if score < current_best_kb[0]:
                current_best_kb = (score, kb)
                print("New best kb with score", score)
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
        # We deep copy here so that the other 3/4 generate below don't have to be
        next_population = [
            copy.deepcopy(config)
            for _, config in scored_population[: len(scored_population) // 4]
        ]
        random.shuffle(scored_population)
        while len(next_population) < POPULATION_SIZE:
            _, kb = scored_population.pop()
            mutate(kb)
            next_population.append(kb)
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


def calculate_kb_score_no_shortcuts(
    kb: Keyboard,
    freq_list: FreqList,
    best: float,
    scores_cache: dict[Keyboard, dict[str, tuple[float, float]]],
    finger_freq_evaluator: FingerFreqEvaluator,
    sfb_evaluator: SFBSFSEvaluator,
    discomfort_evaluator: DiscomfortEvaluator,
    redirect_evaluator: RedirectEvaluator,
    scores_dict: dict[Categories, list[float]],
) -> float:

    score = 0
    finger_freq_evaluator.set_kb(kb)
    sfb_evaluator.set_kb(kb)
    discomfort_evaluator.set_kb(kb)
    redirect_evaluator.set_kb(kb)
    score = 0
    redirect_sum = 0
    for trigram in redirect_evaluator.trigrams.items():
        redirect_sum += redirect_evaluator.evaluate_trigram(trigram)
    redirect_rank = get_rank(redirect_sum, scores_dict[Categories.REDIRECT])
    score += constants_rank.REDIRECT_WEIGHT * redirect_rank
    sfb_sum = 0
    discomfort_sum = 0
    for bigram in sfb_evaluator.bigrams.items():
        sfb_sum += sfb_evaluator.evaluate_bigram(bigram)
        discomfort_sum += discomfort_evaluator.evaluate_bigram(bigram)
    discomfort_rank = get_rank(discomfort_sum, scores_dict[Categories.DISCOMFORT])
    sfb_rank = get_rank(sfb_sum, scores_dict[Categories.SFB])
    score += constants_rank.DISCOMFORT_WEIGHT * discomfort_rank
    score += constants_rank.SFB_WEIGHT * sfb_rank
    sfs_sum = 0
    for i, skipgrams in enumerate(sfb_evaluator.skipgrams):
        for skip_str, skip_freq in skipgrams.items():
            sfs_sum += sfb_evaluator.evaluate_skipgram((skip_str, skip_freq, i))
    sfs_rank = get_rank(sfs_sum, scores_dict[Categories.SFS])
    score += constants_rank.SFS_WEIGHT * sfs_rank

    fingerfreq_sum = finger_freq_evaluator.evaluate_finger_frequencies_MSE(
        constants_rank.GOAL_FINGER_FREQ
    )
    fingerfreq_rank = get_rank(fingerfreq_sum, scores_dict[Categories.FINGERFREQ])
    score += constants_rank.FINGER_FREQ_WEIGHT * fingerfreq_rank
    inaccuracy_sum = evaluate_inaccuracy(kb, freq_list, sys.float_info.max)
    inaccuracy_rank = get_rank(inaccuracy_sum, scores_dict[Categories.INACCURACY])
    score += constants_rank.INACCURACY_WEIGHT * inaccuracy_rank
    return score