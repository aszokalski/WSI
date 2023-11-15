from solver.solver import Solver


class EvoSolver(Solver):
    """A solver for evolutionary algorithms."""

    def __init__(self, strategy: Strategy):
        self.strategy = strategy

    def get_parameters(self):
        return {"strategy": self.strategy}

    def solve(self, problem: Function, x0: np.ndarray, *args, **kwargs) -> Result:
        """Solves the given problem for given initial solution."""
