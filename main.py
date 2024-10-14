import random

from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax


class GraDo15(TwoPlayerGame):
    def __init__(self, players):
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
        """Sprawdza, czy gracz przegrał: jeśli suma liczb gracza wynosi 15."""
        if self.current_player == 1:
            return sum(self.player1_moves) == 15
        else:
            return sum(self.player2_moves) == 15

    def is_over(self):
        """Gra kończy się, gdy któryś z graczy uzyska sumę 15 lub wykonano wszystkie ruchy."""
        return self.lose() or (len(self.player1_moves) == 3 and len(self.player2_moves) == 3)

    def scoring(self):
        """Funkcja oceny dla AI"""
        return 0 if self.lose() else 100

    def show(self):
        """Wyświetlanie aktualnego stanu gry."""
        print(f"Dostępne liczby: {self.pile}")
        print(f"Player 1: {self.player1_moves}, suma: {sum(self.player1_moves)}")
        print(f"Player 2: {self.player2_moves}, suma: {sum(self.player2_moves)}")


ai_algo = Negamax(6)

if __name__ == '__main__':
    gra = GraDo15([Human_Player(), AI_Player(ai_algo)])
    gra.play()
