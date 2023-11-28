from abc import abstractmethod, ABC
from wsilib.utils.function import Function
from wsilib.solver.result import Result
import numpy as np


class Solver(ABC):
    """A solver. It may be initialized with some hyperparameters."""

    @abstractmethod
    def get_parameters(self):
        """Returns a dictionary of hyperparameters"""
        ...

    @abstractmethod
    def solve(
        self, problem: Function, x0: np.ndarray = None, *args, **kwargs
    ) -> Result:
        """
        A method that solves the given problem for given initial solution.
        It may accept or require additional parameters.
        Returns the solution and may return additional info.
        """
        ...
