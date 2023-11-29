from player import Player, RandomPlayer
from game import TwoPlayerGame, TicTacToe
from typing import Literal, Callable, List, Tuple
import warnings
import numpy as np


class MiniMaxPlayer(Player):
    """A player that uses the minimax algorithm to determine its moves."""

    def __init__(
        self,
        game: TwoPlayerGame,
        name: Literal[1, 0],
        depth: int = None,
        heuristic: Callable[[Tuple, Literal[1, 0]], int] = None,
    ):
        if (depth is None) ^ (heuristic is None):
            raise ValueError("Must specify both depth and heuristic or neither")

        self._depth = depth
        self._heuristic = heuristic
        self._reward_table = {}

        super().__init__(game, name)

    def _best_move(self, possible_moves: List, moving: int) -> Tuple:
        """Returns the best move from the given list of possible moves.
        self._reward_table must be populated before calling this method.
        """

        def move_reward(move: List) -> int:
            try:
                return self._reward_table[tuple(move)]
            except KeyError:
                raise ValueError("Move not in reward table")

        return tuple(
            (max if moving == 1 else min)(
                possible_moves,
                key=move_reward,
            )
        )

    def _minimax(self, state: List | Tuple, depth: int, moving: int) -> int:
        """Recursively populates self._reward_table with the reward for each move for each player."""
        state = tuple(state)
        terminal, winner = self._game.is_terminal(state)
        if terminal:
            if winner is None:
                return 0

            return np.inf * (1 if (winner == 1) else -1)

        elif self._depth and depth == 0:
            return self._heuristic(state, moving)

        possible_moves = self._game.get_moves(state, moving)
        if len(possible_moves) == 0:
            warnings.warn(
                "Game.is_terminal() missed a terminal state. Given Game class is faulty. Continuing..."
            )
            return 1 - moving
        for move in possible_moves:
            self._reward_table[move] = self._minimax(move, depth - 1, 1 - moving)

        best_move = self._best_move(possible_moves, moving)

        return self._reward_table[best_move]

    def get_move(self) -> List:
        """Uses the minimax algorithm to determine the best move."""
        super().get_move()

        self._reward_table = {}
        self._minimax(self._game.state, self._depth if self._depth else 0, self._name)

        possible_moves = self._game.get_moves(self._game.state, self._name)

        if len(possible_moves) == 0:
            raise ValueError("No possible moves. Game is lost.")
        return self._best_move(possible_moves, self._name)


if __name__ == "__main__":
    game = TicTacToe(size=3)

    def heuristic(state: Tuple, turn: Literal[1, 0]) -> int:
        state = list(state)
        """Returns the heuristic value of the given state for the given player."""
        for i in range(len(state)):
            if state[i] is None:
                state[i] = 0
            elif state[i] == turn:
                state[i] = 1
            else:
                state[i] = -1
        matrix = np.array(state).reshape((3, 3))

        point_matrix = np.array(
            [
                [3, 2, 3],
                [2, 4, 2],
                [3, 2, 3],
            ]
        )
        return np.sum(point_matrix * matrix)

    players = [
        RandomPlayer(game, 0),
        MiniMaxPlayer(game, 1),
    ]

    p = 0
    while True:
        print(game.state, game.turn)
        result = game.make_move(players[p].get_move())
        p = 1 - p
        if result[0]:
            print(game.state, game.turn)
            break

    print("Game over. Winner:", result[1])
