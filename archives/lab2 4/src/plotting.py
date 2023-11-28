import matplotlib.pyplot as plt
from typing import List
from wsilib.algorithms.evo.evo import EvoResult
import numpy as np


def plot_results(results: List[EvoResult], mean: bool = True):
    all_f_values = [
        [iteration.f_value for iteration in result.history] for result in results
    ]

    mean_f_values = np.mean(all_f_values, axis=0)

    for i, result in enumerate(results):
        n_iters = [iteration.n_iter for iteration in result.history]
        f_values = [iteration.f_value for iteration in result.history]
        time_running = result.history[-1].time_running
        plt.plot(
            n_iters,
            f_values,
            label=f"Execution {i + 1}, Time: {time_running} s",
            alpha=0.7,
        )

    if mean:
        plt.plot(n_iters, mean_f_values, label="Mean", linestyle="--", color="black")

    plt.xlabel("Number of Iterations")
    plt.ylabel("Function Value")
    plt.title("Iteration History with Mean Line" if mean else "Iteration History")
    plt.legend()
    plt.show()


def plot_cities(cities, path):
    x = [city[0] for city in cities]
    y = [city[1] for city in cities]

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, color="blue")

    for i in range(len(path) - 1):
        start_city = path[i]
        end_city = path[i + 1]
        plt.plot(
            [x[start_city], x[end_city]], [y[start_city], y[end_city]], color="red"
        )

    plt.plot([x[path[-1]], x[path[0]]], [y[path[-1]], y[path[0]]], color="red")

    plt.title("Cities and Path")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()
