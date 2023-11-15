from itertools import pairwise
from algorithms.evo.population import Population
from algorithms.evo.individual import IndividualType
from dataclasses import dataclass
from typing import Callable
import numpy as np
import random


@dataclass
class Operation:
    name: str
    function: Callable[[Population, IndividualType], Population]


class GeneticOperations:
    """Class containing genetic operations."""

    @staticmethod
    def mutation(mutation_param: float):
        """Mutate individuals with probability mutation_param"""

        def mutation_method(
            population: Population, individualType: IndividualType
        ) -> Population:
            """Mutate population."""
            return np.array(
                [
                    individualType.mutate(individual, mutation_param)
                    for individual in population
                ]
            )

        return Operation(f"mutation({mutation_param})", mutation_method)

    @staticmethod
    def single_point_crossover():
        """Crossover two consecutive individuals at random point."""

        def crossover_method(
            population: Population, individualType: IndividualType
        ) -> Population:
            """Crossover population."""
            new_population = []
            i = random.randint(0, len(individualType.n_genes) - 1)
            for individualA, individualB in pairwise(population):
                new_population.append(individualA[:i] + individualB[i:])
                new_population.append(individualB[:i] + individualA[i:])
            return np.array(new_population)

        return Operation("single_point_crossover", crossover_method)
