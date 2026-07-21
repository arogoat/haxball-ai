# Egocentryczna obserwacja 2v2 - serce wspoldzielonej polityki wieloagentowej.
# Ta sama siec steruje kazdym z 4 graczy, ale ZA KAZDYM RAZEM widzi swiat z
# perspektywy TEGO gracza: pilka wzgledem mnie, moj kolega wzgledem mnie, dwaj
# przeciwnicy wzgledem mnie. Dzieki temu:
#   - jeden "mozg" gra na kazdej pozycji i po obu stronach (lustro dla niebieskich),
#   - koordynacja/podania pojawiaja sie emergentnie (kazdy wie, ze kolega "mysli
#     tak samo"),
#   - kazdy tick meczu to 4 przyklady uczace zamiast jednego.
#
# Uklad wektora (dlugosc 24):
#   pilka:        poz(2) pred(2)
#   poprzednia akcja: (3)
#   ja:           poz(2) pred(2)
#   kolega:       poz wzgl. mnie(2) pred(2)
#   przeciwnik A: poz wzgl. mnie(2) pred(2)   (blizszy pilki)
#   przeciwnik B: poz wzgl. mnie(2) pred(2)   (dalszy)
# Przeciwnicy sortowani wg odleglosci od PILKI (nie od id) - siec dostaje
# spojne role ("napastnik/cofniety") niezaleznie od numerow graczy.
from typing import Any, List

import numpy as np
from ursinaxball.modules import PlayerHandler

from haxballgym.utils.common_values import BLUE_TEAM
from haxballgym.utils.gamestates import GameState
from haxballgym.utils.obs_builders import ObsBuilder

from futsal_config import POS_SCALE, VEL_SCALE

OBS_SIZE = 23


class Ego2v2Obs(ObsBuilder):
    def reset(self, initial_state: GameState):
        pass

    def build_obs(
        self, player: PlayerHandler, state: GameState, previous_action: np.ndarray
    ) -> Any:
        mirror = np.array([-1.0, 1.0]) if player.team == BLUE_TEAM else np.array([1.0, 1.0])
        ball = state.ball

        self_pos = player.disc.position * mirror
        self_vel = player.disc.velocity * mirror

        obs: List[float] = []
        # pilka (absolutna, znormalizowana, lustrzana)
        obs += list(ball.position * mirror / POS_SCALE)
        obs += list(ball.velocity * mirror / VEL_SCALE)
        # poprzednia akcja
        obs += list(np.asarray(previous_action, dtype=float))
        # ja (absolutnie - siec musi wiedziec, gdzie na boisku jestem)
        obs += list(self_pos / POS_SCALE)
        obs += list(self_vel / VEL_SCALE)

        teammates = []
        opponents = []
        for other in state.players:
            if other.id == player.id:
                continue
            if other.team == player.team:
                teammates.append(other)
            else:
                opponents.append(other)

        # kolega (wzgledem mnie)
        self._add_relative(obs, teammates[0] if teammates else None, self_pos, mirror)

        # przeciwnicy posortowani wg odleglosci od pilki (spojne role)
        opponents.sort(key=lambda o: float(np.linalg.norm(o.disc.position - ball.position)))
        self._add_relative(obs, opponents[0] if len(opponents) > 0 else None, self_pos, mirror)
        self._add_relative(obs, opponents[1] if len(opponents) > 1 else None, self_pos, mirror)

        return np.asarray(obs, dtype=np.float32)

    def _add_relative(self, obs: List[float], other, self_pos: np.ndarray, mirror: np.ndarray):
        if other is None:
            obs += [0.0, 0.0, 0.0, 0.0]
            return
        pos = other.disc.position * mirror
        vel = other.disc.velocity * mirror
        obs += list((pos - self_pos) / POS_SCALE)
        obs += list(vel / VEL_SCALE)
