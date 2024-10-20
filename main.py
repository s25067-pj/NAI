#zasady gry opisane są pod poniższym linkiem
#https://www.gamesforyoungminds.com/blog/2018/5/25/fifteen

#Autorzy kodu
#Paulina Debis s25067
#Krystian Jank s24586

#Gra oparta na dokumentacji https://pypi.org/project/easyAI/ oraz wbudowanej biblioteki do kombinacji cyfr jako tuple

from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax
from itertools import combinations


class GraDo15(TwoPlayerGame):
    def __init__(self, players):
        """Konstruktor przyjmujący graczy jako argumenty"""
        self.players = players
        self.pile = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.player1_moves = []
        self.player2_moves = []
        self.current_player = 1

    def possible_moves(self):
        """Zwraca dostępne ruchy"""
        return [str(n) for n in self.pile]

    def make_move(self, move):
        """Wykonaj ruch: usuń liczbę ze stosu i przypisz ją do aktualnego gracza."""
        move = int(move)
        self.pile.remove(move)

        if self.current_player == 1:
            self.player1_moves.append(move)
        else:
            self.player2_moves.append(move)

    def lose(self):
        """Sprawdza, czy gracz przegrał: jeśli suma trzech liczb gracza wynosi 15."""

        def check_sum(moves):
            """Sprawdza czy istnieje taka kombinacja która suma cyfr to 15
               jako parametr przyjmuje wykoanne ruchy"""
            return any(sum(combo) == 15 for combo in combinations(moves, 3))

        return check_sum(self.player1_moves) if self.current_player == 1 else check_sum(self.player2_moves)

    def is_over(self):
        """Gra kończy się, gdy któryś z graczy przegra lub wykonano 3 ruchy przez obu graczy."""
        return self.lose() or (len(self.player1_moves) == 3 and len(self.player2_moves) == 3)

    def scoring(self):
        """AI ocenia swoje szanse na wygraną.
        Więcej punktów, jeśli jest bliżej wygranej (suma trzech liczb = 15),
        mniej punktów, jeśli przeciwnik zbliża się do wygranej."""

        def score_moves(moves):
            """Funkcja pomocnicza która szuka największej kombinacji dającej sume 15 lub mniejsza,jesli nie ma takiej
            kombinacji zwroc zero (bład z ValueError)"""
            return max([sum(combo) for combo in combinations(moves, 3) if sum(combo) <= 15], default=0)

        if self.current_player == 1:
            return -score_moves(self.player2_moves)
        else:
            return score_moves(self.player1_moves)

    def show(self):
        """Wyświetlanie aktualnego stanu gry."""
        print(f"Dostępne liczby: {self.pile}")
        print(f"Player 1: {self.player1_moves}, suma: {sum(self.player1_moves)}")
        print(f"Player 2: {self.player2_moves}, suma: {sum(self.player2_moves)}")


ai_algo = Negamax(8)

gra = GraDo15([Human_Player(), AI_Player(ai_algo)])

gra.play()
