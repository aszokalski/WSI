from matplotlib import pyplot as plt
import numpy as np


def print_tic_tac_toe(board):
    n = int(np.sqrt(len(board)))  # Assuming n^2 length tuple
    board = np.array(board).reshape((n, n))

    for i in range(n):
        for j in range(n):
            if board[i, j] == 1:
                print(" O ", end="")
            elif board[i, j] == 0:
                print(" X ", end="")
            else:
                print("   ", end="")

            if j < n - 1:
                print("|", end="")

        print()

        if i < n - 1:
            print("---+" * (n - 1) + "---")
    print("\n")


def plot_results(results, player1_name, player2_name):
    fig, ax = plt.subplots()

    depth_labels = [f"Depth {depth}" for depth in results.keys()]
    bar_width = 0.2
    index = np.arange(len(depth_labels))

    player1_wins = [result[0] for result in results.values()]
    player2_wins = [result[1] for result in results.values()]
    draws = [result[2] for result in results.values()]

    ax.bar(index - bar_width, player1_wins, bar_width, label=player1_name + " wins")
    ax.bar(index, player2_wins, bar_width, label=player2_name + " wins")
    ax.bar(index + bar_width, draws, bar_width, label="Draws")

    ax.set_xticks(index)
    ax.set_xticklabels(depth_labels)
    ax.set_xlabel("Depth")
    ax.set_ylabel("Number of Games")
    ax.set_title(f"{player1_name} vs {player2_name}")
    ax.legend()

    plt.show()
