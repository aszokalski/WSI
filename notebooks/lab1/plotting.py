from imports import Function, Domain, Result
import autograd.numpy as np
import matplotlib.pyplot as plt

DEFAULT_LINSPACE = np.linspace(-5, 5, 20)
FIG_SIZE = (9, 5)


def plot_function(
    function: Function, path: np.ndarray = None, domain: np.ndarray = None
):
    """Function that plots the objective function and its gradient

    Args:
        function (Function): objective function to plot
        path (np.ndarray, optional): path to plot. Defaults to None.
        domain (np.ndarray, optional): domain of the function [X range, (Y range)]. Defaults to None.

    Raises:
        ValueError: if function dimension is greater than 2
        ValueError: if path dimension does not match function dimension
    """

    if function.dim > 2:
        raise ValueError("Function dimension is too high to plot")
    if function.dim == 2:
        if path is not None:
            if path.shape[1] != function.dim:
                raise ValueError("Path dimension does not match function dimension")
        _plot_function_3d(function, path, domain)
    else:
        if path is not None:
            if path.ndim > 2:
                raise ValueError("Path dimension does not match function dimension")
        _plot_function_2d(function, path, domain)

    plt.show()


def _plot_function_3d(function: Function, path: np.ndarray = None, domain=None):
    """helper function to plot 3d function and its gradient"""
    figure = plt.figure(figsize=FIG_SIZE)
    figure.suptitle(f"Function: {function.name}")

    x = np.array(domain[0] if domain is not None else DEFAULT_LINSPACE)
    y = np.array(domain[1] if domain is not None else DEFAULT_LINSPACE)
    X, Y = np.meshgrid(x, y)
    Z = function.f(np.array([X, Y]))
    G = function.gradient(np.array([X, Y]))
    U = G[0]
    V = G[1]

    ax_3d = figure.add_subplot(121, projection="3d")
    ax_3d.plot_surface(X, Y, Z, cmap="summer", label="objective function", alpha=0.6)

    # plot path
    if path is not None:
        path_X, path_Y = path.T
        ax_3d.plot(
            path_X,
            path_Y,
            function.f([path_X, path_Y]),
            color="red",
            label="path",
            marker="o",
            markersize=4,
        )

    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")

    ax_3d.set_title("3D Plot")
    ax_3d.legend()

    ax_3d.view_init(20, 170)

    ax_2d = figure.add_subplot(122)
    ax_2d.contour(X, Y, Z, 20, cmap="summer")
    ax_2d.quiver(X, Y, U, V, color="black", pivot="middle", label="gradient")

    # plot path
    if path is not None:
        path_X, path_Y = path.T
        ax_2d.plot(path_X, path_Y, color="red", label="path", marker="o", markersize=4)

    ax_2d.set_xlabel("X")
    ax_2d.set_ylabel("Y")

    ax_2d.set_title("2D Contour Plot with Gradient Vectors ")
    ax_2d.legend()


def _plot_function_2d(function: Function, path: np.ndarray = None, domain=None):
    """helper function to plot 2d function and its gradient"""
    figure = plt.figure(figsize=FIG_SIZE)
    figure.suptitle(f"Function: {function.name}")

    ax = figure.add_subplot(111)
    X = np.array(domain[0] if domain is not None else DEFAULT_LINSPACE)

    ax.plot(X, function.f(X), color="green", label="objective function")
    ax.plot(X, function.gradient(X), color="lightgreen", label="gradient")

    # plot path
    if path is not None:
        ax.plot(
            path, function.f(path), color="red", label="path", marker="o", markersize=4
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()


conditions_list = [
    ["warunek 1", "warunek 2", "warunek 3"],
    ["warunek 1", "warunek 2"],
    ["warunek 1"],
]
step_sizes = [1, 2, 3, 4]
results = []


def plot_single_result(res: Result, f: Function, domain: Domain, title: str = None):
    """Plots single result from solver"""
    if title:
        f.name = title
    path = np.array([iteration.x for iteration in res.history])
    plot_function(f, domain=domain, path=path)


def plot_experiment_results(results: list, conditions_list: list, step_sizes: list):
    """Plots results of experiment"""
    X, Y = np.meshgrid(np.arange(len(conditions_list)), step_sizes)
    results = np.array(results)
    results = results.reshape(X.shape)

    min_y, min_x = np.unravel_index(np.argmin(results), results.shape)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    contour = ax.pcolormesh(X, Y, results, cmap="viridis")
    # contour = ax.contourf(X, Y, results, cmap="viridis")

    fig.colorbar(contour, ax=ax)

    ax.scatter(
        min_x,
        step_sizes[min_y],
        color="red",
        marker="o",
        s=100,
        label="Smallest Minimum",
    )

    xtick_labels = [i + 1 for i in range(len(conditions_list))]
    ax.set_xticks(np.arange(len(conditions_list)), xtick_labels)

    # displaying conditions list below the plot
    plt.figtext(
        0,
        -0.1,
        "Conditions List: \n"
        + "\n".join(
            [
                f"{i+1}. " + ", ".join(conditions)
                for i, conditions in enumerate(conditions_list)
            ]
        ),
        wrap=True,
        horizontalalignment="left",
    )
    ax.legend()
    ax.set_xlabel("Conditions List (Refer to Legend)")
    ax.set_ylabel("Step Sizes")
    ax.set_title("Minimum Value of Function for different hyperparameters", pad=20)
    ax.set_yticks(step_sizes)

    plt.show()
