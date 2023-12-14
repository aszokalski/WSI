# Lab 3

## Struktura

* `/src` - kod źródłowy specyficzny dla laboratorium (np. eksperymenty, wizualizacje)
* `/lib` - kod źródłowy wspólny dla wszystkich laboratoriów (np. implementacja algorytmów, funkcje pomocnicze,
  interfejsy). Jest on instalowany jako pakiet Pythona o nazwie `wsilib`.
* `raport.ipynb` - raport z laboratorium w formacie Jupyter Notebook
* `raport.pdf` - raport z laboratorium w formacie PDF
* `requirements.txt` - lista zależności

## Setup

`pip install -r requirements.txt`

## Uruchomienie

Prezentacja działania zaimplementowanych algorytmów znajduje się w [raporcie](raport.ipynb)

## Przykład użycia

W folderze `src` znajdują się skrypty [`mnist.py`](src/mnist.py) i [`moons.py`](src/moons.py), które prezentują
przykładowe użycie klasyfikatora SVM

## Dokumentacja

### `wsilib.classifier.classifiers.Classifier`

Interfejs klasyfikatora

#### Metody

* `train(X, y, epochs)` - dopasowuje klasyfikator do danych
* `predict(X)` - przewiduje etykiety dla danych
* `score(X, y)` - zwraca dokładność klasyfikatora na danych

### `wsilib.algorithms.svm.SVC(C, kernel, learning_rate)`

Zgodny z interfejsem `Classifier` klasyfikator SVM

#### Parametry

* `kernel` - jądro używane przez klasyfikator
* `C` - parametr regularyzacji
* `learning_rate` - współczynnik uczenia

#### Jądra

* `SVC.polynomial_kernel(degree)` - jądro wielomianowe
* `SVC.gaussian_kernel(gamma)` - jądro gaussowskie

### `wsilib.classifier.classifiers.OneToRestClassifier(BinaryClassifier)`

Klasyfikator pozwalający na klasyfikację wieloklasową za pomocą klasyfikatorów binarnych metodą jeden do reszty

Spełnia interfejs `Classifier`

#### Użycie

```python
classifier = OneToRestClassifier(SVC(...))
classifier.train(X, y)
classifier.predict(X)
...
```
