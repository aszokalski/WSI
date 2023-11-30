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
