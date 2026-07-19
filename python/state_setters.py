# Losowe pozycje startowe - port jittera z wersji JS (env-1v1.js), mocniejszy.
# Bez tego kazdy epizod zaczyna sie identycznie (kickoff), a deterministyczna
# polityka rozgrywa w kolko JEDEN scenariusz: rush czerwonego z kickoffu.
# Losowy start wymusza pelen repertuar sytuacji, w tym OBRONE (np. przeciwnik
# zaczyna z pilka pod nasza bramka - jedyny sposob unikniecia kary z shapingu
# i -10 za gola to interwencja). To wlasnie tak uczy sie obrony - a nie przez
# jawna nagrode "za obronienie strzalu", ktora latwo farmic (bot celowo
# dopuszczalby do strzalow, zeby je bronic).
import random

from ursinaxball import Game

from haxballgym.utils.state_setters import StateSetter

# Wymiary boiska classic: pilka odbija sie na x=+-370, y=+-170.
FIELD_X = 340.0
FIELD_Y = 150.0
BALL_X = 320.0
BALL_Y = 130.0


class RandomState(StateSetter):
    """Z prawdopodobienstwem kickoff_prob zwykly kickoff (bot musi tez umiec
    normalne rozpoczecie), w pozostalych przypadkach pelna losowosc pozycji."""

    def __init__(self, kickoff_prob: float = 0.2):
        super().__init__()
        self.kickoff_prob = kickoff_prob

    def reset(self, game: Game, save_recording: bool):
        game.reset(save_recording)
        if random.random() < self.kickoff_prob:
            return

        ball = game.stadium_game.discs[0]
        ball.position[0] = random.uniform(-BALL_X, BALL_X)
        ball.position[1] = random.uniform(-BALL_Y, BALL_Y)
        # niewielka losowa predkosc startowa pilki - lamie tez "bariere kickoffu"
        # (silnik czeka na ruch pilki), jak w wersji JS
        ball.velocity[0] = random.uniform(-1.5, 1.5)
        ball.velocity[1] = random.uniform(-1.5, 1.5)

        for player in game.players:
            player.disc.position[0] = random.uniform(-FIELD_X, FIELD_X)
            player.disc.position[1] = random.uniform(-FIELD_Y, FIELD_Y)
            player.disc.velocity[0] = 0.0
            player.disc.velocity[1] = 0.0
