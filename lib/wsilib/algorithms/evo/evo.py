from wsilib.solver.solver import Solver
from dataclasses import dataclass, field

from wsilib.algorithms.evo.individual import IndividualType, BinaryIndividualType
from wsilib.algorithms.evo.population import Population
from wsilib.algorithms.evo.succession_metods import SuccessionMethod, SuccessionMethods
from wsilib.algorithms.evo.genetic_operations import GeneticOperations, Operation
from wsilib.solver.stop_conditions import StopConditions, Condition
from wsilib.algorithms.evo.selection_methods import SelectionMethods, SelectionMethod
from wsilib.utils.function import Function
from wsilib.solver.result import Result
from wsilib.solver.iteration import Iteration
import time
import numpy as np

TIME_RUNNING_ACCURACY = 4


@dataclass
class EvoIteration(Iteration):
    population: Population = field(repr=False)


@dataclass
class EvoResult(Result):
    population: Population = field(repr=False)


@dataclass
class EvoSolver(Solver):
    """A solver for evolutionary algorithms."""

    individual_type: IndividualType = field(
        default_factory=lambda: BinaryIndividualType()
    )
    population_size: int = 10
    selection_method: SelectionMethod = field(
        default_factory=lambda: SelectionMethods.tournament_selection(2)
    )
    genetic_operations: list[Operation] = field(
        default_factory=lambda: [
            GeneticOperations.mutation(0.1),
            GeneticOperations.single_point_crossover(),
        ]
    )
    succession_method: SuccessionMethod = field(
        default_factory=lambda: SuccessionMethods.generational_succession()
    )
    stop_conditions: list[Condition] = field(
        default_factory=lambda: [StopConditions.max_iterations(100)]
    )

    def get_parameters(self):
        return {
            "individual_type": self.individual_type,
            "population_size": self.population_size,
            "selection_method": self.selection_method,
            "genetic_operations": self.genetic_operations,
            "succession_method": self.succession_method,
            "stop_conditions": self.stop_conditions,
        }

    def __get_satisfied_condition(self, iteration: Iteration):
        """Returns a satisfied condition name or None."""
        for condition in self.stop_conditions:
            if condition.function(iteration):
                return condition.name
        return None

    def solve(
        self,
        problem: Function,
        x0: np.ndarray = None,
        log: bool = False,
        log_interval_time: int = 1,
    ) -> Result:
        history = []
        n_iter = 0
        start_time = time.process_time()

        log_counter = 0

        population = Population.random(self.individual_type, self.population_size)

        x = x0
        if x0 is None:
            x = population[0]
        x_fitness = problem.f(x)

        while True:
            iteration = EvoIteration(
                x=x,
                population=population,
                f_value=np.linalg.norm(problem.f(x)),
                n_iter=n_iter,
                time_running=round(
                    time.process_time() - start_time, TIME_RUNNING_ACCURACY
                ),
            )

            if log and (
                iteration.time_running > log_interval_time * log_counter
                or log_interval_time == 0
            ):
                print(iteration)
                log_counter += 1

            satisfied_condition = self.__get_satisfied_condition(iteration)

            if satisfied_condition is not None:
                return EvoResult(
                    x0=x0,
                    x=iteration.x,
                    population=population,
                    f_value=iteration.f_value,
                    n_iter=iteration.n_iter,
                    time_running=iteration.time_running,
                    stop_condition=satisfied_condition,
                    history=history,
                )

            history.append(iteration)

            new_population = self.selection_method.function(population, problem)

            mutated_population = new_population
            for operation in self.genetic_operations:
                mutated_population = operation.function(
                    mutated_population, self.individual_type
                )

            population = sorted(population, key=problem.f)
            mutated_population = sorted(mutated_population, key=problem.f)

            curr_x = mutated_population[0]
            curr_x_fitness = problem.f(curr_x)

            if curr_x_fitness < x_fitness:
                x = curr_x
                x_fitness = curr_x_fitness

            population = self.succession_method.function(population, mutated_population)

            n_iter += 1
