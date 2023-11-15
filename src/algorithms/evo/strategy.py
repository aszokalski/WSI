from dataclasses import dataclass, field
from typing import Callable

from algorithms.evo.individual import IndividualType, BinaryIndividualType
from algorithms.evo.population import Population
from algorithms.evo.succession_metods import SuccessionMethod, SuccessionMethods
from solver.stop_conditions import StopConditions, Condition
from algorithms.evo.selection_methods import SelectionMethods, SelectionMethod


@dataclass
class Strategy:
    individual_type: IndividualType = field(default_factory=BinaryIndividualType)
    population_size: int = 10
    selection_method: SelectionMethod = field(
        default=SelectionMethods.tournament_selection(2)
    )
    genetic_operations: list[Callable[[Population], Population]] = field(
        default_factory=list
    )
    succession_method: SuccessionMethod = field(
        default=SuccessionMethods.generational_succession()
    )
    stop_conditions: list[Condition] = field(
        default_factory=lambda: [StopConditions.max_iterations(100)]
    )
