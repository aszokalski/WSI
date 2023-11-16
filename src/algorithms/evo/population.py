import numpy as np
from algorithms.evo.individual import IndividualType


class Population(np.ndarray):
    @classmethod
    def random(cls, individual_type: IndividualType, size: int):
        """Generate a random population."""
        return np.array([individual_type.generate_random() for _ in range(size)])
