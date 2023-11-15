from dataclasses import dataclass
from typing import Tuple
from abc import ABC, abstractmethod
import random


@dataclass(frozen=True)
class IndividualType(ABC):
    """Interface of a class representing individual type."""

    discrete: bool
    bounds: Tuple[float, float]
    n_genes: int

    @abstractmethod
    def generate_random(self):
        """Generate a random individual."""
        pass


class BinaryIndividualType(IndividualType):
    """Class representing binary individual type."""

    def __init__(self, n_genes: int):
        """Initialize binary individual type."""
        super().__init__(discrete=True, bounds=(0, 1), n_genes=n_genes)

    def generate_random(self):
        """Generate a random binary individual."""
        return [random.randint(0, 1) for _ in range(self.n_genes)]


class UnitRangeIndividualType(IndividualType):
    """Class representing individual type with genes in [0, 1] range."""

    def __init__(self, n_genes: int):
        """Initialize normal individual type."""
        super().__init__(discrete=False, bounds=(-1, 1), n_genes=n_genes)

    def generate_random(self):
        """Generate a random normal individual."""
        return [random.uniform(0, 1) for _ in range(self.n_genes)]
