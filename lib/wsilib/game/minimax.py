from player import Player
from game import TwoPlayerGame, TicTacToe
from typing import Literal, Callable, List, Tuple
import warnings


class MiniMaxPlayer(Player):
    """A player that uses the minimax algorithm to determine its moves."""

    def __init__(
        self,
        game: TwoPlayerGame,
        name: Literal[1, 0],
        depth: int = None,
        heuristic: Callable[[List], int] = None,
    ):
        if bool(depth) ^ bool(heuristic):
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
        if self._game.is_terminal(state):
            return 1 - moving
        elif self._depth and depth == 0:
            return self._heuristic(state)

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
    game = TicTacToe()
    players = [MiniMaxPlayer(game, 0), MiniMaxPlayer(game, 1)]

    p = 0
    while True:
        print(game.state, game.turn)
        result = game.make_move(players[p].get_move())
        p = 1 - p
        if result is not None:
            print(game.state, game.turn)
            break

    print("Game over. Winner:", result)
