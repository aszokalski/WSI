from dataclasses import dataclass, field
from typing import Callable
import autograd.numpy as np
from autograd import elementwise_grad


@dataclass
class Function:
    """Class representing function."""

    f: Callable[[np.array], np.array] = field()
    gradient: Callable[[np.array], np.array] = field(default=None)
    dim: int = field(kw_only=True)
    name: str = field(default=None)

    def __post_init__(self):
        if self.gradient is None:
            self.gradient = elementwise_grad(self.f)


class Domain(np.ndarray):
    """Class representing domain of function."""

    def __new__(cls, x, y=None, *args):
        dims = [dim for dim in [x, y, *args] if dim is not None]
        obj = np.asarray(dims).view(cls)
        return obj

    def generate_random_vector(self):
        """Generate random vector from domain."""
        return np.array([np.random.choice(dim) for dim in self])

    def contains(self, x):
        """Check if x is within domain ranges"""
        for i, dim in enumerate(self):
            if x[i] < dim[0] or dim[-1] < x[i]:
                return False
        return True
