from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List, Literal, Tuple
from functools import cache


@dataclass
class TwoPlayerGame(ABC):
    """Abstract class for a two-player game."""

    turn: Literal[1, 0] = 0
    state: List = field(default=None, init=False)

    def __post_init__(self, start_state: List = None):
        if start_state is not None:
            if not self.is_valid_state(start_state):
                raise ValueError("Invalid start state")
            self.state = start_state
        else:
            self.state = self.get_start_state()

    @abstractmethod
    def get_start_state(self) -> List:
        """Returns the start state of the game"""
        ...

    @abstractmethod
    def is_valid_state(self, state: List) -> bool:
        """Returns True if the given state is valid, False otherwise"""
        ...

    @cache
    @abstractmethod
    def get_moves(self, state: Tuple, turn: Literal[0, 1]) -> List[Tuple]:
        """Returns a list of possible moves from the given state"""
        ...

    @cache
    @abstractmethod
    def is_terminal(self, state: Tuple) -> bool:
        """Returns True if the given state is terminal, False otherwise"""
        ...

    def make_move(self, next_state: List) -> Literal[1, 0, None]:
        """Makes a move from the current state to the next state.
        If the move is invalid, raises a ValueError.
        Returns the winner if the game is over, None otherwise.
        """
        if next_state not in self.get_moves(self.state, self.turn):
            raise ValueError("Invalid move")
        self.state = next_state
        result = self.turn if self.is_terminal(self.state) else None
        self.turn = 1 - self.turn
        return result


class TicTacToe(TwoPlayerGame):
    """Tic-tac-toe game class.
    The state is represented as a list of size^2 values,
    where None represents a blank square, 0 an X, and 1 an O.
    """

    def __init__(self, size: int = 3, start_state: List = None):
        self._size = size
        super().__post_init__(start_state)

    def get_start_state(self) -> List:
        return [None] * self._size**2

    def is_valid_state(self, state: List) -> bool:
        return len(state) == self._size**2 and all(v in [None, 0, 1] for v in state)

    def get_moves(self, state: Tuple, turn: Literal[0, 1]) -> List[Tuple]:
        return [
            tuple(state[:i]) + tuple([turn]) + tuple(state[i + 1 :])
            for i, v in enumerate(state)
            if v is None
        ]

    def is_terminal(self, state: Tuple) -> bool:
        if None not in state:
            return True

        horizontals = [
            state[i * self._size : (i + 1) * self._size] for i in range(self._size)
        ]

        verticals = [state[i :: self._size] for i in range(self._size)]

        diagonals = [
            state[:: self._size + 1],
            state[self._size - 1 : self._size**2 - 1 : self._size - 1],
        ]

        return any(
            all(v == line[0] and v is not None for v in line)
            for line in horizontals + verticals + diagonals
        )


# if __name__ == "__main__":
#     game = TicTacToe(size=3)
#     print(game.state)
#     move = game.get_moves()[0]
#     print(move)
#     game.make_move(move)
#     move = game.get_moves(state=)[0]
#     game.state = [None, 0, 1, 0, 1, 1, 1, 1, 0]
#     print(game.is_terminal())
