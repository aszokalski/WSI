from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from wsilib.algorithms.nn.nn import NNC, Layer, OutputLayer

digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = NNC(
    layers=[
        Layer(64),
        Layer(32),
        Layer(16),
        OutputLayer(10),
    ],
    learning_rate=1e-2,
)

clf.train(X_train, y_train, epochs=100)

result = clf.score(X_test, y_test)
result.plot_confusion_matrix()
print(result)
