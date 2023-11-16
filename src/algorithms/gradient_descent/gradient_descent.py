import time
from typing import Callable
import autograd.numpy as np
from autograd import elementwise_grad
from dataclasses import dataclass, field
from utils.function import Function
from utils.domain import Domain
from solver.stop_conditions import StopConditions, Condition
from solver.iteration import Iteration
from solver.result import Result
from solver.solver import Solver

# utils


TIME_RUNNING_ACCURACY = 4


@dataclass
class GradientDescentFunction(Function):
    """Class representing a function with a gradient."""

    gradient: Callable[[np.array], np.array] = field(default=None)

    def __post_init__(self):
        if self.gradient is None:
            self.gradient = elementwise_grad(self.f)


@dataclass
class GradientDescentIteration(Iteration):
    gradient_value: np.ndarray = field()


@dataclass
class GradientDescentResult(Result):
    gradient_value: np.ndarray = field()


class GradientDescentStopConditions(StopConditions):
    """Class containing static methods for stop conditions."""

    @staticmethod
    def min_gradient(min_gradient: float):
        """Stop the experiment when gradient is smaller than min_gradient."""

        def stop_condition(iteration: GradientDescentIteration):
            return np.linalg.norm(iteration.gradient_value) <= min_gradient

        return Condition(stop_condition, f"min_gradient({min_gradient})")


# algorithm


@dataclass
class GradientDescentSolver(Solver):
    step_size: float = field(default=0.01)
    stop_conditions: list = field(
        default_factory=lambda: [GradientDescentStopConditions.max_iterations(100)]
    )

    def get_parameters(self):
        """Returns a dictionary of hyperparameters"""
        return {
            "step_size": self.step_size,
            "stop_conditions": [con.name for con in self.stop_conditions],
        }

    def __get_satisfied_condition(self, iteration: Iteration):
        """Returns a satisfied condition name or None."""
        for condition in self.stop_conditions:
            if condition.function(iteration):
                return condition.name
        return None

    def solve(
        self,
        problem: GradientDescentFunction,
        x0: np.ndarray,
        domain: Domain,
        log: bool = False,
        log_interval_time: int = 1,
    ) -> GradientDescentResult:
        """
        A method that solves the given problem for given initial solution.
        It may accept or require additional parameters.
        Returns the solution and may return additional info.
        """
        x = x0
        history = []
        n_iter = 0
        start_time = time.process_time()

        log_counter = 0

        while True:
            iteration = GradientDescentIteration(
                x=x,
                f_value=np.linalg.norm(problem.f(x)),
                gradient_value=problem.gradient(x),
                n_iter=n_iter,
                time_running=round(
                    time.process_time() - start_time, TIME_RUNNING_ACCURACY
                ),
            )

            if log and iteration.time_running > log_interval_time * log_counter:
                print(iteration)
                log_counter += 1

            satisfied_condition = self.__get_satisfied_condition(iteration)

            if not domain.contains(x):
                satisfied_condition = "X_OUT_OF_DOMAIN"

            if satisfied_condition is not None:
                return GradientDescentResult(
                    x0=x0,
                    x=iteration.x,
                    f_value=iteration.f_value,
                    gradient_value=iteration.gradient_value,
                    n_iter=iteration.n_iter,
                    time_running=iteration.time_running,
                    stop_condition=satisfied_condition,
                    history=history,
                )

            history.append(iteration)
            x = x - self.step_size * iteration.gradient_value
            n_iter += 1
