from dataclasses import dataclass, field
import autograd.numpy as np
from wsilib.solver.iteration import Iteration
from abc import ABC


@dataclass
class Result(ABC):
    """Class representing result of solver."""

    n_iter: int = field()
    time_running: int = field()
    x: np.ndarray = field()
    stop_condition: str = field()
    f_value: float = field()

    x0: np.ndarray = field(repr=False)
    history: list[Iteration] = field(repr=False)
