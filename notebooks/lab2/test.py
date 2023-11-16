# all explicilty imported modules
from imports import (
    Population,
    BinaryIndividualType,
    TSPIndividualType,
    UnitRangeIndividualType,
    GeneticOperations,
    SelectionMethods,
    SuccessionMethods,
    EvoIteration,
    EvoResult,
    EvoSolver,
    StopConditions,
    Function,
)
import numpy as np
import math

from experiments import experiment, params_to_label

# f = Function(
#     lambda x: np.sum(x[:50]),
#     dim=2,
# )

# cities = [
#     [35, 51],
#     [113, 213],
#     [82, 280],
#     [322, 340],
#     [256, 352],
#     [160, 24],
#     [322, 145],
#     [12, 349],
#     [282, 20],
#     [241, 8],
#     [398, 153],
#     [182, 305],
#     [153, 257],
#     [275, 190],
#     [242, 75],
#     [19, 229],
#     [303, 352],
#     [39, 309],
#     [383, 79],
#     [226, 343],
# ]


# def generate_loss_function(city_coordinates):
#     def loss_function(path):
#         total_distance = 0
#         # Calculate distance for each pair of consecutive cities in the path
#         for i in range(len(path) - 1):
#             current_city = path[i]
#             next_city = path[i + 1]

#             x1, y1 = city_coordinates[current_city]
#             x2, y2 = city_coordinates[next_city]

#             distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#             total_distance += distance

#         # Add distance from the last city back to the starting city to complete the loop
#         first_city = path[0]
#         last_city = path[-1]

#         x1, y1 = city_coordinates[last_city]
#         x2, y2 = city_coordinates[first_city]

#         distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#         total_distance += distance

#         return total_distance

#     return Function(loss_function, dim=len(city_coordinates))


# solver = EvoSolver(
#     individual_type=TSPIndividualType(n_genes=len(cities)),
#     population_size=1000,
#     selection_method=SelectionMethods.tournament_selection(2),
#     genetic_operations=[
#         GeneticOperations.mutation(0.1),
#         GeneticOperations.tsp_crossover(0.5),
#     ],
#     succession_method=SuccessionMethods.generational_succession(),
#     stop_conditions=[
#         StopConditions.max_iterations(10000),
#     ],
# )

# solver.solve(generate_loss_function(cities), log=True, log_interval_time=0

exp = experiment(
    params={
        "population_size": [10, 100, 1000],
        "selection_method": [
            SelectionMethods.tournament_selection(2),
            SelectionMethods.tournament_selection(3),
        ],
        "genetic_operations": [
            [
                GeneticOperations.mutation(),
            ],
            [
                GeneticOperations.mutation(),
                GeneticOperations.tsp_crossover(0.7),
            ],
            [
                GeneticOperations.mutation(),
                GeneticOperations.tsp_crossover(0.5),
            ],
        ],
        "succession_method": [
            SuccessionMethods.generational_succession(),
            SuccessionMethods.elitism_succession(1),
        ],
        "stop_conditions": [
            StopConditions.max_iterations(100),
            StopConditions.max_time(10),
        ],
    },
    n_sets=10,
)


for params, results, progress in exp:
    label = params_to_label(params)
    print(progress)
    print(label)
    for result in results:
        print(result.history)
