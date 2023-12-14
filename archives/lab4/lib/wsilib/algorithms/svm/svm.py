from typing import Callable

from wsilib.classifier.classifiers import (
    BinaryClassifier,
    EpochLog,
    BinaryPrediction,
)
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from wsilib.utils.np_cache import np_cache
from sklearn.decomposition import PCA


@dataclass
class SVC(BinaryClassifier):
    _n_training = 0

    @staticmethod
    def polynomial_kernel(degree=3):
        def f(c: float, X: np.ndarray, Y: np.ndarray):
            return (X @ Y.T + c) ** degree

        return f

    @staticmethod
    def gaussian_kernel(sigma=0.5):
        def f(_, X: np.ndarray, Y: np.ndarray):
            return np.exp(
                -np.linalg.norm(X[:, None] - Y[None, :], axis=-1) ** 2
                / (2 * sigma ** 2)
            )

        return f

    @np_cache
    def _cached_kernel(self, c: float, X: np.ndarray, Y: np.ndarray):
        return self.kernel(c, X, Y)

    C: float = 1.0
    kernel: Callable[[float, np.ndarray, np.ndarray], float] = gaussian_kernel()
    learning_rate: float = 0.01

    def __post_init__(self):
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._alphas: np.ndarray | None = None
        self._bias: float = 0.0
        self._alpha_y: np.ndarray | None = None

    def _train_setup(self, X: np.ndarray, y: np.ndarray) -> None:
        self._n_training += 1
        self._X = X
        self._y = y
        self._alphas = np.zeros(self._X.shape[0])
        self._bias = 0.0

    def _train_epoch(self, epoch: int) -> EpochLog:
        kernel_X_X = self._cached_kernel(self.C, self._X, self._X)

        gradient = (
                np.ones(self._X.shape[0]) - self._y * kernel_X_X @ self._alphas * self._y
        )

        self._alphas += gradient * self.learning_rate

        self._alphas = np.clip(self._alphas, 0, self.C)

        loss = np.sum(self._alphas) - 0.5 * np.sum(
            np.outer(self._alphas, self._alphas) * kernel_X_X
        )
        return EpochLog(epoch, loss)

    def _train_teardown(self) -> None:
        self._bias = np.mean(
            self._y
            - np.sum(
                self._alphas * self._y * self.kernel(self.C, self._X, self._X),
                axis=1,
            )
        )

        self._alpha_y = self._alphas * self._y

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        return self._alpha_y @ self.kernel(self.C, self._X, X) + self._bias

    def predict(self, X) -> BinaryPrediction:
        return BinaryPrediction(
            class_name=np.sign(self._decision_function(X)),
            confidence=self._decision_function(X),
        )

    def __hash__(self):
        """A hash function that allows caching."""
        return hash(self._n_training)

    def plot_decision_boundary(self):
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(self._X)

        plt.scatter(
            X_2d[:, 0],
            X_2d[:, 1],
            c=self._y,
            s=50,
            alpha=0.5,
        )
        plot_axes = plt.gca()
        x_limits = plot_axes.get_xlim()
        y_limits = plot_axes.get_ylim()

        # create grid to evaluate model
        x_grid_points = np.linspace(x_limits[0], x_limits[1], 30)
        y_grid_points = np.linspace(y_limits[0], y_limits[1], 30)
        grid_y, grid_x = np.meshgrid(y_grid_points, x_grid_points)
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
        grid_points_full_dim = pca.inverse_transform(grid_points)

        decision_function_values = self._decision_function(
            grid_points_full_dim
        ).reshape(grid_x.shape)

        # plot decision boundary and margins
        plot_axes.contour(
            grid_x,
            grid_y,
            decision_function_values,
            colors=["b", "g", "r"],
            levels=[-1, 0, 1],
            alpha=0.5,
            linestyles=["--", "-", "--"],
            linewidths=[2.0, 2.0, 2.0],
        )

        # highlight the support vectors
        plot_axes.scatter(
            X_2d[:, 0][self._alphas > 0.0],
            X_2d[:, 1][self._alphas > 0.0],
            s=50,
            linewidth=1,
            facecolors="none",
            edgecolors="k",
        )

        plt.show()
