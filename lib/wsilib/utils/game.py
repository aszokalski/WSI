from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List, Literal
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

    @abstractmethod
    @cache
    def get_moves(self, state: List = None) -> List:
        """Returns a list of possible moves from the given state"""
        ...

    @abstractmethod
    @cache
    def is_terminal(self, state: List = None) -> bool:
        """Returns True if the given state is terminal, False otherwise"""
        ...

    def make_move(self, next_state: List) -> Literal[1, 0, None]:
        """Makes a move from the current state to the next state.
        If the move is invalid, raises a ValueError.
        Returns the winner if the game is over, None otherwise.
        """
        if next_state not in self.get_moves():
            raise ValueError("Invalid move")
        self.state = next_state
        self.turn = 1 - self.turn
        return 1 - self.turn if self.is_terminal(self.state) else None


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

    def get_moves(self, state: List = None) -> List:
        if state is None:
            state = self.state
        return [
            state[:i] + [self.turn] + state[i + 1 :]
            for i, v in enumerate(state)
            if v is None
        ]

    def is_terminal(self, state: List = None) -> bool:
        if state is None:
            state = self.state
        return (
            any(
                all(state[i] == state[i + j] for j in [1, 2])
                for i in [0, 3, 6, 0, 1, 2, 0, 2]
            )
            or 0 not in state
        )


if __name__ == "__main__":
    game = TicTacToe(size=3)
    print(game.state)
    move = game.get_moves()[0]
    print(move)
    game.make_move(move)
    move = game.get_moves()[0]
    print(move)
