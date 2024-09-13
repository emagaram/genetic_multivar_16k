import json
import os

from score_categories import Categories

input_dir = "./create_datasets_input"
output_dir="./create_datasets_output"
all_scores:dict[Categories, list[float]] = {category: [] for category in Categories}

for filename in os.listdir(input_dir):
    file_path = os.path.join(input_dir, filename)
    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            for category in Categories:
                if file_path.find(category.value) != -1:
                    scores = json.load(file)
                    all_scores[category].extend(scores)
                    all_scores[category].sort()
    else:
        raise Exception(f"Invalid file path {file_path}")

directory_path = os.path.join(output_dir)
os.makedirs(directory_path, exist_ok=True)
for category, scores in all_scores.items():
    filename = os.path.join(directory_path, f"{category.value}.json")
    with open(filename, "w") as file:
        file.write(f"{json.dumps(scores)}")