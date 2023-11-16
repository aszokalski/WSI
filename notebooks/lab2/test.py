# all explicilty imported modules
from imports import (
    Population,
    BinaryIndividualType,
    UnitRangeIndividualType,
    GeneticOperations,
    SelectionMethods,
    SuccessionMethods,
    EvoIteration,
    EvoResult,
    EvoSolver,
    StopConditions,
    Function,
)


f = Function(
    lambda x: x[0] ** 2 + x[1] ** 2,
    dim=2,
)

solver = EvoSolver(
    individual_type=UnitRangeIndividualType(10),
    population_size=100,
    selection_method=SelectionMethods.tournament_selection(2),
    genetic_operations=[
        GeneticOperations.mutation(0.1),
        GeneticOperations.single_point_crossover(),
    ],
    succession_method=SuccessionMethods.generational_succession(),
    stop_conditions=[
        StopConditions.max_iterations(100),
    ],
)

solver.solve(f, log=True)
