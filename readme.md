# LAB WSI

## Biblioteka laboratoryjna

Laboratorium posługuje się wspólną boblioteką, w której zaimplementowane są badane algorytmy i funckje pomocnicze. Biblioteka ta jest instalowana jako pakiet Pythona o nazwie `wsilib` i znajduje się w katalogu `lib`.

### Instalacja

Jest ona instalowana wraz z innymi zależnościami z pliku `requirements.txt` za pomocą polecenia:

## Setup

```bash
pip install -r requirements.txt
```

## Budowanie archiwum alboratorium

Aby zbudować archiwum z kodem źródłowym laboratorium należy wykonać polecenie:

```bash
python build_lab_archive.py --lib ./lib --lab ./notebooks/<lab_name>
```

gdzie `<lab_name>` to nazwa katalogu z laboratorium (np. `lab1`).

Powstanie wówczas archiwum `<lab_name>.zip` w katalogu `archives` zawierające kod źródłowy laboratorium.
(W archiwum znajdą się jedynie pliki dodane do gita).
