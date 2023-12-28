import copy

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from typing import Literal, List, Callable


@dataclass
class TrainingScore:
    """A dataclass that stores training score."""

    accuracy: float
    precision: float = None
    recall: float = None
    f1_score: float = None
    confusion_matrix: np.ndarray = field(default=None, repr=False)

    def plot_confusion_matrix(self):
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.xticks(range(len(self.confusion_matrix)))
        plt.yticks(range(len(self.confusion_matrix)))

        for i in range(len(self.confusion_matrix)):
            for j in range(len(self.confusion_matrix)):
                plt.text(j, i, self.confusion_matrix[i, j], ha="center", va="center")

        plt.imshow(self.confusion_matrix)
        plt.colorbar()
        plt.show()


@dataclass
class EpochLog:
    epoch: int
    loss: float


@dataclass
class ClassifierLog:
    epochs: List[EpochLog]

    def plot_loss(self, title=None):
        plt.plot(
            [epoch.epoch for epoch in self.epochs],
            [epoch.loss for epoch in self.epochs],
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        if title is not None:
            plt.title(title)
        plt.show()


@dataclass
class Prediction:
    """A dataclass that stores prediction."""

    class_name: np.ndarray
    confidence: np.ndarray


@dataclass
class BinaryPrediction(Prediction):
    class_name: Literal[-1, 1]


class Classifier(ABC):
    """A classifier. It may be initialized with some hyperparameters."""

    @abstractmethod
    def train(self, X: np.array, y: np.array, epochs: int = 100) -> None:
        ...

    # @np_cache
    @abstractmethod
    def predict(self, X) -> Prediction:
        ...

    @abstractmethod
    def score(self, X: np.array, y: np.array) -> TrainingScore:
        ...


class BinaryClassifier(Classifier, ABC):
    """A binary classifier. It may be initialized with some hyperparameters."""

    @abstractmethod
    def _train_setup(self, X: np.ndarray, y: np.ndarray) -> None:
        """A method that is called before training. It may be used to initialize some variables."""
        ...

    @abstractmethod
    def _train_epoch(self, epoch: int) -> EpochLog:
        """A method that is called each epoch. It should return the training loss."""
        ...

    def _train_teardown(self) -> None:
        """A method that is called after training. It may be used to clean up some variables."""
        pass

    def predict(self, X) -> BinaryPrediction:
        ...

    def train(self, X: np.array, y: np.array, epochs: int = 100) -> ClassifierLog:
        logs = []
        self._train_setup(X, y)
        for epoch in range(epochs):
            log = self._train_epoch(epoch)
            logs.append(log)
            print(log)
        self._train_teardown()
        return ClassifierLog(logs)

    def score(self, X: np.array, y: np.array) -> TrainingScore:
        y_pred = self.predict(X).class_name
        accuracy = np.mean(y_pred == y)
        tp = np.sum((y_pred == 1) & (y == 1))
        fp = np.sum((y_pred == 1) & (y == -1))
        fn = np.sum((y_pred == -1) & (y == 1))

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        return TrainingScore(accuracy, precision, recall, f1_score)

    @abstractmethod
    def plot_decision_boundary(self):
        ...


class OneToRestClassifier(Classifier):
    def __init__(self, classifier: BinaryClassifier):
        self.classifier = classifier
        self.classifiers: dict[int, BinaryClassifier] = {}

    def train(self, X: np.array, y: np.array, epochs: int = 100) -> None:
        self.classifiers = {}
        for class_name in np.unique(y):
            y_binary = np.where(y == class_name, 1, -1)
            # copy classifier
            classifier = copy.deepcopy(self.classifier)

            classifier.train(X, y_binary, epochs).plot_loss(class_name)
            classifier.plot_decision_boundary()
            self.classifiers[class_name] = classifier

    def predict(self, X: np.array) -> np.array:
        results = []
        for x in X:
            x = np.array(x)
            predictions = {}
            for class_name, classifier in self.classifiers.items():
                predictions[class_name] = classifier.predict(x).confidence
            results.append(max(predictions, key=predictions.get))
        return np.array(results)

    def score(self, X: np.array, y: np.array) -> TrainingScore:
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        confusion_matrix = np.zeros((len(self.classifiers), len(self.classifiers)))
        for i, class_name in enumerate(self.classifiers):
            for j, class_name2 in enumerate(self.classifiers):
                confusion_matrix[i, j] = np.sum(
                    (y_pred == class_name) & (y == class_name2)
                )
        return TrainingScore(accuracy, confusion_matrix=confusion_matrix)
