from abc import ABC
from wsilib.solver.iteration import Iteration
from dataclasses import dataclass
from typing import Callable


@dataclass
class Condition:
    """Class representing condition."""

    function: Callable[[Iteration], bool]
    name: str


class StopConditions(ABC):
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
    def once():
        """Stop the experiment after first iteration."""

        def stop_condition(_):
            return True

        return Condition(stop_condition, "once()")
