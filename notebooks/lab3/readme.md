# Lab 3

## Struktura

* `/src` - kod źródłowy specyficzny dla laboratorium (np. eksperymenty, wizualizacje)
* `/lib` - kod źródłowy wspólny dla wszystkich laboratoriów (np. implementacja algorytmów, funkcje pomocnicze, interfejsy). Jest on instalowany jako pakiet Pythona o nazwie `wsilib`.
* `raport.ipynb` - raport z laboratorium w formacie Jupyter Notebook
* `raport.pdf` - raport z laboratorium w formacie PDF
* `requirements.txt` - lista zależności

## Setup

`pip install -r requirements.txt`

## Uruchomienie

Prezentacja działania zaimplementowanych algorytmów znajduje się w [raporcie](raport.ipynb)

## Przykład użycia
```Python
game = TicTacToe(size=3)

players = [
    RandomPlayer(game, 0),
    MiniMaxAlphaBetaPlayer(game, 1),
]

p = 0
while True:
    print(game.state, game.turn)
    result = game.make_move(players[p].get_move())
    p = 1 - p
    if result[0]:
        print(game.state, game.turn)
        break

print("Game over. Winner:", result[1])
```

## Dokumentacja

### `wsilib.game.TwoPlayerGame(start_state: Optional[List] = None)`

Klasa bazowa dla gier dwuosobowych. Definiuje interfejs, który muszą implementować wszystkie gry dwuosobowe. Niektóre metody abstrakcyjne są cache'owane, co znacznie przyspiesza działanie algorytmów.

Przechowuje:

* `state` - stan gry
* `turn` - numer gracza, który ma teraz wykonać ruch

Implementuje:

* `make_move(self, next_state: List)` - wykonuje ruch `next_state` i zwraca krotkę `(game_over, winner)`, gdzie `game_over` jest wartością logiczną, która mówi, czy gra się zakończyła, a `winner` jest numerem gracza, który wygrał grę. Jeśli `game_over` jest `False`, to `winner` jest równy `None`.

Metody abstrakcyjne:

* `get_start_state(self) -> List` - zwraca stan początkowy gry
* `is_valid_state(self, state: List) -> bool` - sprawdza, czy stan `state` jest poprawny
* __(cached)__ `get_moves(self, state: Tuple, turn: Literal[0,1]) -> List[Tuple]` - zwraca listę możliwych ruchów dla gracza `turn` w stanie `state`
* __(cached)__ `is_terminal(self, state: Tuple) -> Tuple[bool, Literal[1, 0, None]` - zwraca krotkę `(game_over, winner)`, gdzie `game_over` jest wartością logiczną, która mówi, czy gra się zakończyła, a `winner` jest numerem gracza, który wygrał grę. Jeśli `game_over` jest `False`, to `winner` jest równy `None`.

#### `wsilib.game.TicTacToe(size: int, start_state: Optional[List] = None)`

Klasa reprezentująca grę w kółko i krzyżyk. Dziedziczy po `TwoPlayerGame`.

### `wsilib.player.Player(game: TwoPlayerGame, name: Literal[0,1])`

Klasa bazowa dla graczy. Definiuje interfejs, który muszą implementować wszyscy gracze.

Implementuje:

* `get_move(self) -> List` - sprawdza, czy jest kolej gracza i zwraca `self._get_move()`

Metody abstrakcyjne:

* `_get_move(self) -> List` - zwraca ruch gracza

#### `wsilib.player.RandomPlayer(game: TwoPlayerGame, name: Literal[0,1])`

Klasa reprezentująca gracza losowego. Dziedziczy po `Player`.

#### `wsilib.algorithms.minimax.MiniMaxPlayer(...)`

Klasa reprezentująca gracza korzystającego z algorytmu minimax. Dziedziczy po `Player`.

Argumenty:

* `game: TwoPlayerGame` - gra, w którą gra gracz
* `name: Literal[0,1]` - numer gracza
* (opt) `depth: int` - głębokość przeszukiwania drzewa gry
* (opt) `heuristic: Callable[[Tuple, Literal[0,1]], float]` - funkcja heurystyczna `(state, turn)=>reward`

Argumenty `depth` i `heuristic` trzeba podać oba lub żadnego.

#### `wsilib.algorithms.minimax.MiniMaxAlphaBetaPlayer(...)`

Klasa reprezentująca gracza korzystającego z algorytmu minimax z cięciami alfa-beta. Dziedziczy po `MiniMaxPlayer`.