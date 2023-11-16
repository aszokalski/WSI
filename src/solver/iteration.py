from dataclasses import dataclass, field
import autograd.numpy as np
from abc import ABC


@dataclass
class Iteration(ABC):
    """Class representing iteration of solver."""

    n_iter: int = field()
    time_running: int = field(metadata={"unit": "s"})
    x: np.ndarray = field()
    f_value: float = field()
