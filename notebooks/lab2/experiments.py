import itertools
import math
from imports import Function, EvoSolver, TSPIndividualType, EvoResult
import numpy as np

cities = [
    [35, 51],
    [113, 213],
    [82, 280],
    [322, 340],
    [256, 352],
    [160, 24],
    [322, 145],
    [12, 349],
    [282, 20],
    [241, 8],
    [398, 153],
    [182, 305],
    [153, 257],
    [275, 190],
    [242, 75],
    [19, 229],
    [303, 352],
    [39, 309],
    [383, 79],
    [226, 343],
]


def avg_f_value(results: list[EvoResult]):
    return np.mean([result.f_value for result in results])


def params_to_label(params):
    label = "Params:\n"
    for params_name, params_value in params.items():
        if isinstance(params_value, list):
            label += f"    {params_name}:\n"
            for params_value_value in params_value:
                label += f"        {params_value_value.name if hasattr(params_value_value, 'name') else params_value_value}\n"
        else:
            label += f"    {params_name}: {params_value.name if hasattr(params_value, 'name') else params_value}\n"
    return label


def generate_cost_function(city_coordinates):
    def loss_function(path):
        total_distance = 0
        # Calculate distance for each pair of consecutive cities in the path
        for i in range(len(path) - 1):
            current_city = path[i]
            next_city = path[i + 1]

            x1, y1 = city_coordinates[current_city]
            x2, y2 = city_coordinates[next_city]

            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            total_distance += distance

        # Add distance from the last city back to the starting city to complete the loop
        first_city = path[0]
        last_city = path[-1]

        x1, y1 = city_coordinates[last_city]
        x2, y2 = city_coordinates[first_city]

        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        total_distance += distance

        return total_distance

    return Function(loss_function, dim=len(city_coordinates))


def generate_cities(n, max_x, max_y):
    cities = []
    for i in range(n):
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
        cities.append([x, y])
    return cities


def test_sets_generator(n_test_sets, max_x=400, max_y=400):
    yield cities
    for _ in range(n_test_sets - 1):
        yield generate_cities(len(cities), max_x, max_y)


def experiment(params, n_sets: int, log: bool = False):
    global cities
    individual_type = TSPIndividualType(n_genes=len(cities))

    for key in params:
        check = params[key]
        if key == "genetic_operations" or key == "stop_conditions":
            check = check[0]
        if not isinstance(check, list):
            params[key] = [params[key]]

    keys = params.keys()
    values = params.values()
    product = itertools.product(*values)
    total = len(list(product))
    i = 0

    for v in itertools.product(*values):
        params = dict(zip(keys, v))
        solver = EvoSolver(
            individual_type=individual_type,
            population_size=params["population_size"],
            selection_method=params["selection_method"],
            genetic_operations=params["genetic_operations"],
            succession_method=params["succession_method"],
            stop_conditions=params["stop_conditions"],
        )

        results = []
        test_sets = test_sets_generator(n_sets)
        for cities in test_sets:
            loss_function = generate_cost_function(cities)
            result = solver.solve(loss_function, log=log)
            results.append(result)

        yield params, results, f"{i + 1}/{total}"
        i += 1
