from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from wsilib.algorithms.svm.svm import SVC
from wsilib.classifier.classifiers import OneToRestClassifier

digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = OneToRestClassifier(
    SVC(C=1.0, kernel=SVC.gaussian_kernel(sigma=0.5), learning_rate=1e-2)
)

clf.train(X_train, y_train, epochs=100)
result = clf.score(X_train, y_train)
result.plot_confusion_matrix()
print(result)

import matplotlib.pyplot as plt
import numpy as np

n = 1
indices = np.random.choice(X_test.shape[0], n)

images = X_test[indices]
labels = y_test[indices]
predictions = clf.predict(images)

for i in range(n):
    plt.imshow(images[i].reshape(8, 8), cmap=plt.cm.gray_r, interpolation="nearest")
    plt.title(f"Prediction: {predictions[i]}, Label: {labels[i]}")
    plt.show()
