import autograd.numpy as np


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
