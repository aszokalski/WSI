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

        return jacobian_matrix


@dataclass
class Layer:
    """A layer of a neural network."""

    size: int = field()
    activation_function: ActivationFunction = field(default=Sigmoid())
    init_method: Literal["random", "zeros"] = field(default="random")
    _learning_rate: float = field(default=0.1, init=False)
    _weights: np.ndarray = field(default=None, init=False)

    def __post_init__(self):
        self._value = np.zeros(self.size)

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
        self._value = self.activation_function(self._weights.T @ x)
        return self._value

    def backpropagate(self, d_next_activation: np.ndarray) -> np.ndarray:
        d_weights = self._value.T @ d_next_activation
        self._weights -= d_weights * self._learning_rate
        return (
                self._weights @ d_next_activation
        ) * self.activation_function.derivative(self._value)


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
        for x, y in zip(X, y):
            pred = self._nn(x)
            loss = (y - pred) ** 2
            self._nn.backpropagate(loss)
        return EpochLog(epoch, loss)

    def train(self, X: np.array, y: np.array, epochs: int = 100) -> ClassifierLog:
        logs = []
        for epoch in range(epochs):
            log = self._train_epoch(epoch, X, y)
            logs.append(log)
            print(log)
        return ClassifierLog(logs)

    def score(self, X: np.array, y: np.array) -> TrainingScore:
        y_pred = self.predict(X).class_name
        accuracy = np.mean(y_pred == y)
        tp = np.sum((y_pred == 1) & (y == 1))
        tn = np.sum((y_pred == -1) & (y == -1))
        fp = np.sum((y_pred == 1) & (y == -1))
        fn = np.sum((y_pred == -1) & (y == 1))

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        return TrainingScore(accuracy, precision, recall, f1_score)


if __name__ == "__main__":
    nn = NeuralNetwork(
        layers=[
            Layer(size=5),
            OutputLayer(size=2),
        ],
        learning_rate=5,
    )

    x = np.array([1, 1, 1, 1, 1])
    y = np.array([0, 1])

    for _ in range(1):
        print(nn(x))
        # r squared
        loss = (y - nn(x)) ** 2
        print("Loss: ", loss)
        nn.backpropagate(loss)

    print(nn(x))
