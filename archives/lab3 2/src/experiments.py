from typing import List, Tuple, Callable
from wsilib.game.player import Player
from wsilib.game.game import TwoPlayerGame
from src.plotting import plot_results


def experiment(
    game: TwoPlayerGame, player1: Player, player2: Player, num_games: int
) -> None:
    """Plays a number of games between two players and prints the results."""
    player1_wins = 0
    player2_wins = 0
    draws = 0
    for _ in range(num_games):
        game.reset()
        while True:
            if game.turn == 0:
                move = player1.get_move()
            else:
                move = player2.get_move()
            result = game.make_move(move)
            if result[0]:
                break
        if result[1] == 0:
            player1_wins += 1
        elif result[1] == 1:
            player2_wins += 1
        else:
            draws += 1
    return player1_wins, player2_wins, draws


def run_experiments(
    game: TwoPlayerGame,
    rival_classes: List[Tuple[Player]],
    depths: List[int],
    heuristic: Callable,
    num_games: int,
) -> None:
    """Runs a number of experiments between players and prints the results."""
    for class1, class2 in rival_classes:
        title = f"{class1.__name__} vs {class2.__name__} @ depth {depths}"
        print(title)
        results = {}
        for depth in depths:
            game.reset()
            player1 = class1(game, 0, depth, heuristic)
            player2 = class2(game, 1, depth, heuristic)
            player1_wins, player2_wins, draws = experiment(
                game, player1, player2, num_games
            )

            results[depth] = (player1_wins, player2_wins, draws)

        plot_results(
            results,
            class1.__name__,
            class2.__name__,
        )
