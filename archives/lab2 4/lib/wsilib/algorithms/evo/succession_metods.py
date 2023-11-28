from wsilib.algorithms.evo.population import Population
from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class SuccessionMethod:
    name: str
    function: Callable[[Population, Population], Population]


class SuccessionMethods:
    """Class containing methods of succession.
    Static methods of this class are used to create SuccessionMethod objects.
    These methods work on sorted populations (by fitness function, descending).
    """

    @staticmethod
    def generational_succession():
        def selection_method(
            previous_population: Population, mutated_population: Population
        ) -> Population:
            return mutated_population

        return SuccessionMethod("generational_succession", selection_method)

    @staticmethod
    def elitism_succession(n_elites: int = 1):
        def selection_method(
            previous_population: Population, mutated_population: Population
        ) -> Population:
            return np.concatenate(
                (previous_population[:n_elites], mutated_population[n_elites:])
            )

        return SuccessionMethod(f"elitism_succession({n_elites})", selection_method)
