from dataclasses import dataclass, field
from collections import namedtuple
from function import Function, Domain
import autograd.numpy as np
import time

TIME_RUNNING_ACCURACY = 4


@dataclass
class Iteration:
    """Class representing iteration of solver."""

    n_iter: int = field()
    time_running: int = field(metadata={"unit": "s"})
    x: np.ndarray = field()
    f_value: float = field()
    gradient_value: np.ndarray = field()


Condition = namedtuple("Condition", ["function", "name"])


class StopConditions:
    """Class containing static methods for stop conditions."""

    @staticmethod
    def max_iterations(max_iter: int):
        """Stop the experiment after max_iter iterations."""

        def stop_condition(iteration: Iteration):
            return iteration.n_iter >= max_iter

        return Condition(stop_condition, f"max_iterations({max_iter})")

    @staticmethod
    def max_time(max_time: int):
        """Stop the experiment after max_time seconds."""

        def stop_condition(iteration: Iteration):
            return iteration.time_running >= max_time

        return Condition(stop_condition, f"max_time({max_time})")

    @staticmethod
    def min_gradient(min_gradient: float):
        """Stop the experiment when gradient is smaller than min_gradient."""

        def stop_condition(iteration: Iteration):
            return np.linalg.norm(iteration.gradient_value) <= min_gradient

        return Condition(stop_condition, f"min_gradient({min_gradient})")

    @staticmethod
    def once():
        """Stop the experiment after first iteration."""

        def stop_condition(_):
            return True

        return Condition(stop_condition, "once()")


@dataclass
class Result:
    """Class representing result of solver."""

    n_iter: int = field()
    time_running: int = field()
    x: np.ndarray = field()
    stop_condition: str = field()
    f_value: float = field()
    gradient_value: np.ndarray = field()

    x0: np.ndarray = field(repr=False)
    history: list[Iteration] = field(repr=False)


@dataclass
class Solver:
    """A solver. It may be initialized with some hyperparameters."""

    step_size: float = field(default=0.01)
    stop_conditions: list = field(
        default_factory=lambda: [StopConditions.max_iterations(100)]
    )

    def get_parameters(self):
        """Returns a dictionary of hyperparameters"""
        return {
            "learning_rate": self.step_size,
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
        problem: Function,
        x0: np.ndarray,
        domain: Domain,
        log: bool = False,
        log_interval_time: int = 1,
    ) -> Result:
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
            iteration = Iteration(
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
                return Result(
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
