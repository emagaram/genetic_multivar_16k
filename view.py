import json
import math
import numpy as np
import matplotlib.pyplot as plt
from score_categories import Categories

scores_dict: dict[str, list[float]] = {}
# Example data
for name in Categories:
    with open(f"./starting_data/{name.value}.json", "r") as file:
        scores_dict[name] = json.load(file)
        # for i, a in enumerate(scores_dict[name]):
        #     scores_dict[name][i] = math.log(scores_dict[name][i])

        # Convert data to a NumPy array for CDF calculation
        data = np.array(scores_dict[name])
        
        # Sort the data
        sorted_data = np.sort(data)
        
        # Compute the CDF
        cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)

        # Plot the CDF
        plt.plot(sorted_data, cdf * 100)  # Multiply by 100 to show percentage
        plt.xlim(0.0075, 0.015)
        # Add labels and title
        plt.xlabel('Value')
        plt.ylabel('Cumulative Percentage')
        plt.title(f'Cumulative Distribution of {name.value.capitalize()} Data')

        # Display the plot
        plt.show()