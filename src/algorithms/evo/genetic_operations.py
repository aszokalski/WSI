from itertools import islice
from algorithms.evo.population import Population
from algorithms.evo.individual import IndividualType
from dataclasses import dataclass
from typing import Callable
import numpy as np
import random


def batched(iterable, n):
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


@dataclass
class Operation:
    name: str
    function: Callable[[Population, IndividualType], Population]


class GeneticOperations:
    """Class containing genetic operations."""

    @staticmethod
    def mutation(mutation_param: float = None):
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

        return Operation(
            f"mutation({mutation_param})" if mutation_param else "mutation",
            mutation_method,
        )

    @staticmethod
    def single_point_crossover():
        """Crossover two consecutive individuals at random point."""

        def crossover_method(
            population: Population, individualType: IndividualType
        ) -> Population:
            """Crossover population."""
            new_population = []
            i = random.randint(0, individualType.n_genes - 1)

            for individualA, individualB in batched(population, 2):
                new_population.append(
                    np.concatenate((individualA[:i], individualB[i:]))
                )
                new_population.append(
                    np.concatenate((individualB[:i], individualA[i:]))
                )
            return np.array(new_population)

        return Operation("single_point_crossover", crossover_method)

    @staticmethod
    def tsp_crossover(alpha: float):
        """Crossover for traveling salesman problem."""

        def crossover_method(
            population: Population, individualType: IndividualType
        ) -> Population:
            """Crossover population."""
            new_population = []
            pivot = int(alpha * individualType.n_genes)

            for individualA, individualB in batched(population, 2):
                (tail_A, head_A) = (individualA[:pivot], individualA[pivot:])
                (tail_B, head_B) = (individualB[:pivot], individualB[pivot:])

                mapping_A = {tail_B[i]: tail_A[i] for i in range(len(tail_A))}
                mapping_B = {tail_A[i]: tail_B[i] for i in range(len(tail_A))}

                for i in range(len(head_A)):
                    while head_A[i] in tail_B:
                        head_A[i] = mapping_A[head_A[i]]
                    while head_B[i] in tail_A:
                        head_B[i] = mapping_B[head_B[i]]

                new_population.append(np.concatenate((tail_A, head_B)))
                new_population.append(np.concatenate((tail_B, head_A)))
            return np.array(new_population)

        return Operation(f"tsp_crossover({alpha})", crossover_method)
