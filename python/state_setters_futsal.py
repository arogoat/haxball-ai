# Losowe pozycje startowe dla futsalu (wieksze boisko niz classic). Analogia do
# state_setters.py, ale zakresy z futsal_config. Losowe starty = pelen repertuar
# sytuacji, w tym obrona; 20% zwyklych kickoffow, zeby bot umial tez normalne
# rozpoczecie.
import random

from ursinaxball import Game
from haxballgym.utils.state_setters import StateSetter

from futsal_config import SPAWN_HALF_X, SPAWN_HALF_Y, BALL_SPAWN_HALF_X, BALL_SPAWN_HALF_Y


class FutsalRandomState(StateSetter):
    def __init__(self, kickoff_prob: float = 0.2):
        super().__init__()
        self.kickoff_prob = kickoff_prob

    def reset(self, game: Game, save_recording: bool):
        game.reset(save_recording)
        if random.random() < self.kickoff_prob:
            return

        ball = game.stadium_game.discs[0]
        ball.position[0] = random.uniform(-BALL_SPAWN_HALF_X, BALL_SPAWN_HALF_X)
        ball.position[1] = random.uniform(-BALL_SPAWN_HALF_Y, BALL_SPAWN_HALF_Y)
        ball.velocity[0] = random.uniform(-2.0, 2.0)
        ball.velocity[1] = random.uniform(-2.0, 2.0)

        for player in game.players:
            player.disc.position[0] = random.uniform(-SPAWN_HALF_X, SPAWN_HALF_X)
            player.disc.position[1] = random.uniform(-SPAWN_HALF_Y, SPAWN_HALF_Y)
            player.disc.velocity[0] = 0.0
            player.disc.velocity[1] = 0.0
