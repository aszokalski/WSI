from abc import ABC, abstractmethod
from typing import List, Literal, Callable
from game import TwoPlayerGame
from functools import wraps


def raise_(ex):
    """Raises the given exception"""
    raise ex


def if_(condition: Callable[["Player"], bool], action: Callable):
    """Decorator that runs the given action if the condition is False."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if condition(args[0]):
                action()

            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


class Player(ABC):
    """Abstract class for a player in a two-player game."""

    def __init__(self, game: TwoPlayerGame, name: Literal[1, 0]):
        self._game = game
        self._name = name

    @if_(
        condition=lambda self: self._game.turn != self._name,
        action=lambda: raise_(ValueError("Not your turn")),
    )
    @abstractmethod
    def get_move(self) -> List:
        """Returns the player's move"""
