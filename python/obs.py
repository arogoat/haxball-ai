# Znormalizowane obserwacje - ta sama struktura co DefaultObs z haxballgym
# (pilka poz+pred, poprzednia akcja, ja poz+pred, przeciwnik poz+pred; wspolrzedne
# lustrzane dla niebieskich, wiec jeden model gra poprawnie po obu stronach),
# ale przeskalowane do ~[-1,1] - PPO duzo lepiej znosi wejscia w tej skali niz
# surowe wspolrzedne rzedu setek.
from typing import Any, List

import numpy as np
from ursinaxball.modules import PlayerHandler

from haxballgym.utils.common_values import BLUE_TEAM
from haxballgym.utils.gamestates import GameState
from haxballgym.utils.obs_builders import ObsBuilder

POS_SCALE = 400.0
VEL_SCALE = 5.0


class NormalizedObs(ObsBuilder):
    def reset(self, initial_state: GameState):
        pass

    def build_obs(
        self, player: PlayerHandler, state: GameState, previous_action: np.ndarray
    ) -> Any:
        ball = state.ball
        mirror = np.array([-1.0, 1.0]) if player.team == BLUE_TEAM else np.array([1.0, 1.0])

        obs: List[np.ndarray] = [
            ball.position * mirror / POS_SCALE,
            ball.velocity * mirror / VEL_SCALE,
            np.asarray(previous_action, dtype=float),
        ]

        self._add_player(obs, player, mirror)
        for other in state.players:
            if other.id == player.id:
                continue
            if other.team == player.team:
                self._add_player(obs, other, mirror)
        for other in state.players:
            if other.id == player.id:
                continue
            if other.team != player.team:
                self._add_player(obs, other, mirror)

        return np.concatenate(obs).astype(np.float32)

    def _add_player(self, obs: List[np.ndarray], player: PlayerHandler, mirror: np.ndarray):
        obs.append(player.disc.position * mirror / POS_SCALE)
        obs.append(player.disc.velocity * mirror / VEL_SCALE)
