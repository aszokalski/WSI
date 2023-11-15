from dataclasses import dataclass, field
from typing import Callable
import autograd.numpy as np
from abc import ABC


@dataclass
class Function(ABC):
    """Class representing function."""

    f: Callable[[np.array], np.array] = field()

    dim: int = field(kw_only=True)
    name: str = field(default=None)


@dataclass
class ErrorFunction(Function):
    """Class representing an error function."""

    pass
