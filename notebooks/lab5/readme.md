# Lab 5

## Struktura

* `/src` - kod źródłowy specyficzny dla laboratorium (np. eksperymenty, wizualizacje)
* `/lib` - kod źródłowy wspólny dla wszystkich laboratoriów (np. implementacja algorytmów, funkcje pomocnicze,
  interfejsy). Jest on instalowany jako pakiet Pythona o nazwie `wsilib`.
* `raport.ipynb` - raport z laboratorium w formacie Jupyter Notebook
* `raport.pdf` - raport z laboratorium w formacie PDF
* `requirements.txt` - lista zależności

## Setup

`Python 3.10` lub nowszy

`pip install -r requirements.txt`

## Uruchomienie

Prezentacja działania zaimplementowanych algorytmów znajduje się w [raporcie](raport.ipynb)

## Przykład użycia

W folderze `src` znajduje się skrypt [`mnist.py`](src/mnist.py) , który prezentuje
przykładowe użycie klasyfikatora NNC

## Dokumentacja

### `wsilib.classifier.classifiers.Classifier`

Interfejs klasyfikatora

#### Metody

* `train(X, y, epochs)` - dopasowuje klasyfikator do danych
* `predict(X)` - przewiduje etykiety dla danych
* `score(X, y)` - zwraca dokładność klasyfikatora na danych

### `wsilib.algorithms.nn.nn.NNC(layers, learning_rate)`

Zgodny z interfejsem `Classifier` klasyfikator SVM

#### Parametry

* `layers` - lista zawierająca obiekty `Layer` reprezentujące warstwy sieci
* `learning_rate` - współczynnik uczenia

### `wsilib.algorithms.nn.nn.Layer(size, activation)`

Obiekt reprezentujący warstwę sieci neuronowej

#### Parametry

* `size` - liczba neuronów w warstwie
* `activation` - funkcja aktywacji warstwy (np. `wsilib.algorithms.nn.nn.Sigmoid`)

