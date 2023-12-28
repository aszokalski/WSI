from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Literal
import numpy as np
from wsilib.utils.np_cache import np_cache
from wsilib.classifier.classifiers import (
    Classifier,
    Prediction,
    EpochLog,
    ClassifierLog,
    TrainingScore,
)
from itertools import pairwise


class ActivationFunction(ABC):
    """An activation function."""

    @np_cache
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the activation function to the input."""
        ...

    @np_cache
    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Calculate the derivative of the activation function."""
        ...


class Sigmoid(ActivationFunction):
    """The sigmoid activation function."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return self(x) * (1 - self(x))


class SoftMax(ActivationFunction):
    """The softmax activation function."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x)
        return e / np.sum(e)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        softmax_values = self(x)
        n = len(softmax_values)

        # Compute the Jacobian matrix
        jacobian_matrix = -np.outer(softmax_values, softmax_values)
        np.fill_diagonal(jacobian_matrix, softmax_values * (1 - softmax_values))

        return np.diag(jacobian_matrix)


@dataclass
class Layer:
    """A layer of a neural network."""

    size: int = field()
    activation_function: ActivationFunction = field(default=Sigmoid())
    init_method: Literal["random", "zeros"] = field(default="random")
    _learning_rate: float = field(default=0.1, init=False)
    _weights: np.ndarray = field(default=None, init=False)

    def __post_init__(self):
        self._input = np.zeros(self.size)

    def set_learning_rate(self, value):
        self._learning_rate = value

    def setup_weights(self, next_size: int):
        if self.init_method == "random":
            self._weights = np.random.randn(self.size, next_size)
        elif self.init_method == "zeros":
            self._weights = np.zeros((self.size, next_size))
        else:
            raise ValueError(f"Invalid init method: {self.init_method}")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the layer to the input."""
        self._input = x
        return self.activation_function(self._weights.T @ x)

    def backpropagate(self, d_next_loss: np.ndarray) -> np.ndarray:
        reuse = np.outer(
            d_next_loss, self.activation_function.derivative(self._input)
        ).T
        d_weights = reuse * self._input[:, np.newaxis]
        self._weights -= d_weights * self._learning_rate

        return np.einsum("ij,ij->i", reuse, self._weights)


@dataclass
class OutputLayer(Layer):
    """The output layer of a neural network."""

    activation_function: ActivationFunction = field(default=SoftMax())

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the layer to the input."""
        self._value = self.activation_function(x)
        return self._value

    def backpropagate(self, loss: np.ndarray) -> np.ndarray:
        return loss * self.activation_function.derivative(self._value)


@dataclass
class NeuralNetwork:
    """A neural network."""

    layers: list[Layer] = field()
    learning_rate: float = field(default=0.1)

    def __post_init__(self):
        for layer, next_layer in pairwise(self.layers):
            layer.setup_weights(next_layer.size)
            layer.set_learning_rate(self.learning_rate)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the neural network to the input."""
        for layer in self.layers:
            x = layer(x)
        return x

    def backpropagate(self, loss: np.ndarray):
        carry = loss
        for layer in reversed(self.layers):
            carry = layer.backpropagate(carry)


class NNC(Classifier):
    """A neural network classifier."""

    def __init__(self, layers: list[Layer], learning_rate: float = 0.1):
        self._nn = NeuralNetwork(layers, learning_rate)

    def predict(self, X: np.ndarray) -> Prediction:
        pred = self._nn(X)
        index = np.argmax(pred)
        return Prediction(
            class_name=index,
            confidence=pred[index],
        )

    def _train_epoch(
            self,
            epoch: int,
            X: np.array,
            y: np.array,
    ) -> EpochLog:
        avg_loss = 0
        for x, y in zip(X, y):
            pred = self._nn(x)

            loss = (y - pred) ** 2
            avg_loss += loss
            self._nn.backpropagate(loss)

        avg_loss /= len(X)
        return EpochLog(epoch, avg_loss)

    def train(self, X: np.array, y: np.array, epochs: int = 100) -> ClassifierLog:
        logs = []
        for epoch in range(epochs):
            log = self._train_epoch(epoch, X, y)
            logs.append(log)
            print(log)
            if epoch % 10 == 0:
                ClassifierLog(logs).plot_loss()
        return ClassifierLog(logs)

    def score(self, X: np.array, y: np.array) -> TrainingScore:
        predictions = np.array([self.predict(x).class_name for x in X])
        true_labels = np.argmax(y, axis=1)

        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(
            true_labels, predictions, average="weighted", zero_division=0
        )
        recall = recall_score(
            true_labels, predictions, average="weighted", zero_division=0
        )
        f1 = f1_score(true_labels, predictions, average="weighted", zero_division=0)
        conf_matrix = confusion_matrix(true_labels, predictions)

        return TrainingScore(accuracy, precision, recall, f1, conf_matrix)
