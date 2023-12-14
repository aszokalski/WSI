from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from wsilib.algorithms.svm.svm import SVC

X, y = make_moons(n_samples=1000, noise=0.1, random_state=0)
y[y == 0] = -1
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

svc = SVC(C=1.0, kernel=SVC.gaussian_kernel(sigma=0.5), learning_rate=1e-2)
svc.train(X_train, y_train, epochs=100)
print(svc.score(X_train, y_train))
svc.plot_decision_boundary()
