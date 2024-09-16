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
from get_stats import get_score_stats
from inaccuracy import InaccuracyEvaluator, InaccuracyMode
from score_categories import Categories
import settings

from finger_freq import FingerFreqEvaluator
from keyboard import Key, Keyboard, MagicKey
from redirect import RedirectEvaluator
from sfb_sfs import SFBSFSEvaluator
from words import create_full_freq_list, create_inaccuracy_freq_list


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


def write_best_kb_to_file(
    base_path,
    performance: dict[str, float],
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
        file.write(get_score_stats(kb, performance))
        file.write("")
        file.write(settings.settings_to_str(space))


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
def mutate(kb: Keyboard, is_good: bool):
    # Idk for this
    loops = random.randint(1, 3) if is_good else random.randint(2, 6)

    for _ in range(loops):
        rand = random.random()
        if rand < 0.3 and len(kb.magic_locations) > 0:
            hand, col, row = random.choice(kb.magic_locations)
            mk: MagicKey = kb.keyboard[hand][col][row]
            mk.mutate()
        # Swap letter
        elif rand < 0.7:
            a_key, a_letter_idx = kb.get_random_letter_kb_index()
            b_key, b_letter_idx = kb.get_random_letter_kb_index()
            a_char = a_key.letters[a_letter_idx]
            b_char = b_key.letters[b_letter_idx]
            a_key.letters = a_key.letters.replace(a_char, b_char, 1)
            b_key.letters = b_key.letters.replace(b_char, a_char, 1)
        # Swap column
        elif rand < 0.75:
            col_a, colb = random.choice(random.choice(kb.keyboard)), random.choice(random.choice(kb.keyboard))
            for key_a, key_b in zip(col_a, colb):
                temp = key_a.letters
                key_a.letters = key_b.letters
                key_b.letters = temp        
        # Move key
        elif rand <= 0.85:
            a_key, _ = kb.get_random_letter_kb_index()
            b_key, _ = kb.get_random_letter_kb_index()
            temp = a_key.letters
            a_key.letters = b_key.letters
            b_key.letters = temp
        # Move letters
        else:
            def can_move_letters_to(moving_letter: str, letters: str):
                return moving_letter == "'" or len(letters) < (
                    2 + 1 if letters.find("'") != -1 else 0
                )

            a_key, a_letter_idx = kb.get_random_letter_kb_index()
            b_key, _ = kb.get_random_letter_kb_index()
            while (
                a_key.letters == b_key.letters
                or len(a_key.letters) == 1
                or not can_move_letters_to(a_key.letters[a_letter_idx], b_key.letters)
            ):
                a_key, a_letter_idx = kb.get_random_letter_kb_index()
                b_key, b_letter_idx = kb.get_random_letter_kb_index()
            a_char = a_key.letters[a_letter_idx]
            a_key.letters = a_key.letters.replace(a_char, "", 1)
            b_key.letters += a_char


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
    best: float,
    scores_cache: dict[Keyboard, dict[str, float]],
    finger_freq_evaluator: FingerFreqEvaluator,
    sfb_evaluator: SFBSFSEvaluator,
    discomfort_evaluator: DiscomfortEvaluator,
    redirect_evaluator: RedirectEvaluator,
    inaccuracy_evaluator: InaccuracyEvaluator,
) -> dict[str, float]:

    score = 0
    if kb not in scores_cache:
        # total_ts = time.time()
        # set_ts = time.time()
        finger_freq_evaluator.set_kb(kb)
        sfb_evaluator.set_kb(kb)
        discomfort_evaluator.set_kb(kb)
        redirect_evaluator.set_kb(kb)
        inaccuracy_evaluator.set_kb(kb)
        # set_te = time.time()
        score = 0

        fingerfreq_sum = (
            finger_freq_evaluator.evaluate_finger_frequencies_max_limit_MAPE()
        )
        score += settings.FINGER_FREQ_WEIGHT * fingerfreq_sum
        # sfb_ts = time.time()
        sfb_sum = sfb_evaluator.evaluate_bigrams_fast(sfb_evaluator.fast_bigrams, 1)
        # sfb_te = time.time()
        score += settings.SFB_WEIGHT * sfb_sum
        sfs_sum = 0
        # sfs_ts = time.time()

        for i, skipgrams in enumerate(sfb_evaluator.fast_skipgrams):
            sfs_sum += sfb_evaluator.evaluate_skipgrams_fast(skipgrams, i, True)
            if score + settings.SFS_WEIGHT * sfs_sum > best:
                break
        # sfs_te = time.time()
        score += settings.SFS_WEIGHT * sfs_sum
        discomfort_sum = 0
        # discomfort_ts = time.time()
        for bigram in sfb_evaluator.bigrams.items():
            discomfort_sum += discomfort_evaluator.evaluate_bigram(bigram)
            if score + settings.DISCOMFORT_WEIGHT * discomfort_sum > best:
                break
        # discomfort_te = time.time()
        score += settings.DISCOMFORT_WEIGHT * discomfort_sum

        # redirect_ts = time.time()
        minimum_failing_redirect_score = (best - score) / settings.REDIRECT_WEIGHT
        redirect_sum = redirect_evaluator.evaluate_fast(minimum_failing_redirect_score)
        # redirect_te = time.time()
        score += settings.REDIRECT_WEIGHT * redirect_sum
        # inaccuracy_ts = time.time()
        minimum_failing_inaccuracy_score = (best - score) / settings.INACCURACY_WEIGHT
        inaccuracy_sum = 0
        if settings.NUM_MAGIC > 0:
            inaccuracy_sum = inaccuracy_evaluator.evaluate_inaccuracy_mode(
                minimum_failing_inaccuracy_score, settings.MODE
            )
        else:
            heuristic = inaccuracy_evaluator.evaluate_inaccuracy_heuristic(
                minimum_failing_inaccuracy_score
            )
            if heuristic > minimum_failing_inaccuracy_score:
                inaccuracy_sum = heuristic
            else:
                inaccuracy_sum = inaccuracy_evaluator.evaluate_inaccuracy_mode(
                    minimum_failing_inaccuracy_score, settings.MODE
                )
        score += settings.INACCURACY_WEIGHT * inaccuracy_sum
        # inaccuracy_te = time.time()
        # total_te = time.time()
        # times:dict[str, float] = {
        #     Categories.DISCOMFORT.value: discomfort_te - discomfort_ts,
        #     Categories.INACCURACY.value: inaccuracy_te - inaccuracy_ts,
        #     Categories.REDIRECT.value: redirect_te - redirect_ts,
        #     Categories.SFS.value: sfs_te - sfs_ts,
        #     Categories.SFB.value: sfb_te - sfb_ts,
        #     "set_kb": set_te - set_ts
        # }
        # times_sorted_lst = sorted(
        #     list((name, time) for name, time in times.items()), key=lambda x: x[1]
        # )
        # total = total_te - total_ts
        # if random.random() < 0.01:
        #     print(f"Total: {1000*total:.3f}ms")
        #     for (name, val) in times_sorted_lst:
        #         print(f"{name.capitalize()}: {100*val/total:.3f}%")
        #     print()
        # print( f"Discomfort pct: {100*(discomfort_te-discomfort_ts)/(total_te-total_ts)}%" )
        scores_cache[kb] = {
            "score": score,
            Categories.FINGERFREQ.value: fingerfreq_sum,
            Categories.INACCURACY.value: inaccuracy_sum,
            Categories.DISCOMFORT.value: discomfort_sum,
            Categories.SFB.value: sfb_sum,
            Categories.SFS.value: sfs_sum,
            Categories.REDIRECT.value: redirect_sum,
        }
    return scores_cache[kb]


def run_simulation(
    iteration_path: str,
    stop_event,
    layout: list[list[int]],
):

    POPULATION_SIZE = 60
    SOLUTION_IMPROVEMNT_DEADLINE = 1000
    PROCESS_ID = str(os.getpid() % 100)
    ERRORS_LOG_FILENAME = f"{PROCESS_ID}_errors.log"
    SCORE_FILE = f"{PROCESS_ID}_best_layout.txt"
    all_time_score = sys.float_info.max
    current_best_score: float = sys.float_info.max
    finger_freq_evaluator = FingerFreqEvaluator(create_full_freq_list())
    # print(finger_freq_evaluator)
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
    population: list[Keyboard] = [Keyboard(layout) for _ in range(POPULATION_SIZE)]          
    generation_count = 1
    total_generation_count = 0
    solution_improvement_count = 0
    while not stop_event.is_set():
        if stop_event.is_set():
            break
        scored_population: list[tuple[float, Keyboard]] = []

        # Called when no new improvements have been made in a while
        if solution_improvement_count > SOLUTION_IMPROVEMNT_DEADLINE:
            scores_cache.clear()
            total_generation_count += generation_count
            write_generations_completed(
                iteration_path, PROCESS_ID, total_generation_count
            )
            print(
                f"Generation died. Total generations: {generation_count}, Reg Score: {current_best_score}"
            )
            generation_count = 1
            solution_improvement_count = 0
            current_best_score = sys.float_info.max
            population = [Keyboard(layout) for _ in range(POPULATION_SIZE)]

        start_time = time.time()  # Start timing the generation
        for kb in population:
            # m s   'cf gp   jn rw   qt ly   |   az ox   ev ku   , bi   . dh

            performance = calculate_kb_score(
                kb,
                current_best_score,
                scores_cache,
                finger_freq_evaluator,
                sfb_evaluator,
                discomfort_evaluator,
                redirect_evaluator,
                inaccuracy_evaluator,
            )
            score = performance["score"]
            scored_population.append((score, kb))

            if score < current_best_score:
                current_best_score = score
                solution_improvement_count = 0
                if score < all_time_score:
                    print("New best all time kb with score", score)
                    all_time_score = score
                    write_best_kb_to_file(
                        iteration_path,
                        performance,
                        kb,
                        SCORE_FILE,
                    )

        # Sort the population based on the score (lower is better)
        scored_population.sort(key=lambda x: x[0])
        avg = sum(score for score, _ in scored_population) / len(scored_population)
        end_time = time.time()
        if settings.PRINT:
            print(
                f"Generation {generation_count} took {end_time - start_time:.2f} seconds, Best score: {current_best_score:.8f}"
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
            score, kb = scored_population.pop()
            mutate(kb, score < avg)
            next_population.append(kb)

        population = next_population


if __name__ == "__main__":
    # Setup stop event

    stop_event = Event()

    # Start the termination monitor in a separate thread
    termination_thread = Thread(target=monitor_for_termination, args=(stop_event,))
    termination_thread.start()

    # Modify your main script logic here
    # int(cpu_count()
    num_processes = 1
    iteration_id = datetime.datetime.now().strftime("%Y_%m_%d_%Hh_%Mm_%Ss")
    processes = []

    # Create a directory for output if it does not exist
    base_path = "optimizer_output"
    iteration_path = os.path.join(base_path, iteration_id)
    os.makedirs(iteration_path, exist_ok=True)

    layout = [[2, 2, 2, 2], [2, 2, 2, 2]]

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


"""

    kb = Keyboard(
        layout,
        [
            [
                [Key("m"), Key("s")],
                [Key("'cf"), Key("gp")],
                [Key("jn"), Key("rw")],
                [Key("qt"), Key("ly")],
            ],
            [
                [Key("az"), Key("ox")],
                [Key("ev"), Key("ku")],
                [Key(","), Key("bi")],
                [Key("."), Key("dh")],
            ],
        ],
    )    
    performance = calculate_kb_score(
        kb,
        current_best_score,
        scores_cache,
        finger_freq_evaluator,
        sfb_evaluator,
        discomfort_evaluator,
        redirect_evaluator,
        inaccuracy_evaluator,
    )
    write_best_kb_to_file(
        "./test1.txt",
        performance,
        kb,
        SCORE_FILE,
    )    
    kb2 = Keyboard(
        layout,
        [
            [
                [Key("mz"), Key("'cy")],
                [Key("px"), Key("lu")],
                [Key("be"), Key("n")],
                [Key("qt"), Key("jr")],
            ],
            [
                [Key("aw"), Key("df")],
                [Key("s"), Key("ko")],
                [Key(","), Key("gi")],
                [Key("."), Key("vh")],
            ],
        ],
    )   
    performance = calculate_kb_score(
        kb2,
        current_best_score,
        scores_cache,
        finger_freq_evaluator,
        sfb_evaluator,
        discomfort_evaluator,
        redirect_evaluator,
        inaccuracy_evaluator,
    )    
    write_best_kb_to_file(
        "./test2.txt",
        performance,
        kb2,
        SCORE_FILE,
    )  

"""