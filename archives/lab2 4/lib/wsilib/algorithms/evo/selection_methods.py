from wsilib.utils.function import Function
from wsilib.algorithms.evo.population import Population
from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class SelectionMethod:
    name: str
    function: Callable[[Population, Function], Population]


class SelectionMethods:
    """Class containing methods of selection."""

    @staticmethod
    def tournament_selection(k: int):
        def selection_method(
            population: Population, cost_function: Function
        ) -> Population:
            """Select k individuals from population using tournament selection."""

            return np.array(
                [
                    min(
                        [
                            population[i]
                            for i in np.random.choice(
                                len(population), size=k, replace=False
                            )
                        ],
                        key=cost_function.f,
                    )
                    for _ in range(len(population))
                ]
            )

        return SelectionMethod(f"tournament_selection({k})", selection_method)
