from dataclasses import dataclass, field
from typing import Callable
import autograd.numpy as np


@dataclass
class Function:
    """Class representing function."""

    f: Callable[[np.array], np.array] = field()

    dim: int = field(kw_only=True)
    name: str = field(default=None)
