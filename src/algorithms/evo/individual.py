from dataclasses import dataclass
from typing import Tuple
from abc import ABC, abstractmethod
import random
import numpy as np


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

    @abstractmethod
    def mutate(self, individual: np.ndarray, mutation_param: float = 0.1):
        """Mutate an individual."""
        pass


class BinaryIndividualType(IndividualType):
    """Class representing binary individual type."""

    def __init__(self, n_genes: int = 10):
        """Initialize binary individual type."""
        super().__init__(discrete=True, bounds=(0, 1), n_genes=n_genes)

    def generate_random(self):
        """Generate a random binary individual."""
        return [random.randint(0, 1) for _ in range(self.n_genes)]

    def mutate(self, individual, mutation_param: float = 0.1):
        """Mutate an individual."""
        return [
            1 - gene if random.random() < mutation_param else gene
            for gene in individual
        ]


class UnitRangeIndividualType(IndividualType):
    """Class representing individual type with genes in [0, 1] range."""

    def __init__(self, n_genes: int = 10):
        """Initialize normal individual type."""
        super().__init__(discrete=False, bounds=(-1, 1), n_genes=n_genes)

    def generate_random(self):
        """Generate a random normal individual."""
        return [random.uniform(0, 1) for _ in range(self.n_genes)]

    def mutate(self, individual, mutation_param: float = 0.1):
        """Mutate an individual using normal distribution."""
        return [
            gene + random.uniform(-mutation_param, mutation_param)
            for gene in individual
        ]


class TSPIndividualType(IndividualType):
    """Class representing individual type for TSP problem."""

    def __init__(self, n_genes: int = 10):
        """Initialize TSP individual type."""
        super().__init__(discrete=True, bounds=(0, n_genes), n_genes=n_genes)

    def generate_random(self):
        """Generate a random TSP individual."""
        return np.random.permutation(self.n_genes)

    def mutate(self, individual, mutation_param: float = 0.1):
        """Mutate an individual."""
        return np.random.permutation(individual)


class DomainIndividualType(IndividualType):
    """Class representing individual type with genes in given range."""

    def __init__(self, n_genes: int = 10, bounds: Tuple[float, float] = (0, 1)):
        super().__init__(discrete=False, bounds=bounds, n_genes=n_genes)

    def generate_random(self):
        """Generate a random individual."""
        return [random.uniform(*self.bounds) for _ in range(self.n_genes)]

    def mutate(self, individual, mutation_param: float = 0.1):
        """Mutate an individual."""
        return [
            min(
                max(
                    gene + random.uniform(-mutation_param, mutation_param),
                    self.bounds[0],
                ),
                self.bounds[1],
            )
            for gene in individual
        ]
