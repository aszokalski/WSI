import sys

sys.path.append("../../src")
sys.path.append("src")

# flake8: noqa
from algorithms.evo.population import Population
from algorithms.evo.individual import (
    BinaryIndividualType,
    UnitRangeIndividualType,
    TSPIndividualType,
    DomainIndividualType,
    IndividualType,
)
from algorithms.evo.genetic_operations import GeneticOperations
from algorithms.evo.selection_methods import SelectionMethods
from algorithms.evo.succession_metods import SuccessionMethods
from algorithms.evo.evo import EvoSolver, EvoIteration, EvoResult
from solver.stop_conditions import StopConditions
from utils.function import Function
from utils.domain import Domain
from solver.result import Result
