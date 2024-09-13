import json
import math

def calculate_z_scores(filename):
    # Load the array of numbers from a JSON file
    with open(filename, 'r') as file:
        numbers = json.load(file)

    # Ensure the input is a list of numbers
    if not isinstance(numbers, list) or not all(isinstance(x, (int, float)) for x in numbers):
        raise ValueError("The file must contain a list of numbers.")

    # Calculate mean
    mean = sum(numbers) / len(numbers)

    # Calculate standard deviation
    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
    std_dev = math.sqrt(variance)

    # Calculate z-scores
    z_scores = [(x - mean) / std_dev for x in numbers]

    # Find max and min z-scores
    max_z = max(z_scores)
    min_z = min(z_scores)

    # Print or return the results
    return {
        "mean": mean,
        "std_dev": std_dev,
        "z_scores": z_scores,
        "max_z_score": max_z,
        "min_z_score": min_z
    }

# Example usage
filename = "./starting_data/inaccuracy.json"
filename2 = "./starting_data/inaccuracy2.json"
def print_results(filename:str):
    print(filename)
    space = "  "
    results = calculate_z_scores(filename)
    print(f"{space}Mean: {results['mean']}")
    print(f"{space}Standard Deviation: {results['std_dev']}")
    print(f"{space}Max Z-Score: {results['max_z_score']}")
    print(f"{space}Min Z-Score: {results['min_z_score']}")
    print()
print_results(filename)
print_results(filename2)