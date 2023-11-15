from abc import abstractmethod, ABC
import autograd.numpy as np
from utils.function import Function
from solver.result import Result


class Solver(ABC):
    """A solver. It may be initialized with some hyperparameters."""

    @abstractmethod
    def get_parameters(self):
        """Returns a dictionary of hyperparameters"""
        ...

    @abstractmethod
    def solve(self, problem: Function, x0: np.ndarray, *args, **kwargs) -> Result:
        """
        A method that solves the given problem for given initial solution.
        It may accept or require additional parameters.
        Returns the solution and may return additional info.
        """
        ...
