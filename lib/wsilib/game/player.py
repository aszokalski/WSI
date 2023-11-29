from abc import ABC, abstractmethod
from typing import List, Literal
from game import TwoPlayerGame
import random


class Player(ABC):
    """Abstract class for a player in a two-player game."""

    def __init__(self, game: TwoPlayerGame, name: Literal[1, 0]):
        self._game = game
        self._name = name

    def get_move(self) -> List:
        """Checks if it's the player's turn and returns the player's move"""
        if self._game.turn != self._name:
            raise ValueError("Not your turn")
        return self._get_move()

    @abstractmethod
    def _get_move(self) -> List:
        """Returns the player's move"""


class RandomPlayer(Player):
    """A player that makes random moves."""

    def _get_move(self) -> List:
        return random.choice(self._game.get_moves(self._game.state, self._name))
