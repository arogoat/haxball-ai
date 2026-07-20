# Shaped reward - port naprawionej wersji z JS (dqn-common-1v1.js), przeskalowany
# do poziomow przyjaznych PPO (gol ~10 zamiast 500). Sklad:
#  - kara za czas (zacheta do konczenia meczu golem, nie remisem)
#  - shaping: zblizanie sie do pilki + pchanie pilki w strone bramki przeciwnika
#  - bonus za dotkniecie pilki (z licznika silnika, nie z heurystyki odleglosci)
#  - gol +/-10, timeout z dodatkowa kara (przez get_final_reward)
# Wnioski z JS przeniesione wprost: proporcje kar/nagrod maja gwarantowac, ze
# przegrana > remis > wygrana NIGDY sie nie odwraca (stara wersja miala kare za
# czas w sumie ~rownowazna stracie gola - bot nie mial powodu bronic bramki).
import numpy as np
from ursinaxball.common_values import TeamID
from ursinaxball.modules import PlayerHandler

from haxballgym.utils.gamestates import GameState
from haxballgym.utils.reward_functions import RewardFunction

GOAL_X = 370.0
GOAL_HALF_HEIGHT = 64.0

STEP_PENALTY = -0.01
APPROACH_WEIGHT = 0.1 / 50.0  # (prev-new) w jednostkach mapy * ta waga
# x2 (bylo 0.002): pchanie pilki w strone bramki ma placic wyraznie, zeby
# aktywna gra ofensywna byla atrakcyjniejsza niz krecenie sie po boisku
BALL_TO_GOAL_WEIGHT = 0.2 / 50.0
TOUCH_REWARD = 0.3
GOAL_REWARD = 10.0
# Bylo -0.6 i doprowadzilo do "tchorzliwej rownowagi": stracony gol -10 vs
# remis -0.6 oznaczal, ze przy dwoch rownych przeciwnikach unikanie gry bylo
# optymalne - po 43M krokow self-play OBAJ boci przestali chodzic do pilki
# (70% pasywnych remisow w eval). Remis musi bolec na tyle, zeby ryzyko
# ataku sie oplacalo, ale wciaz mniej niz stracony gol.
TIMEOUT_PENALTY = -4.0


def _ball_dist_to_goal(ball_pos: np.ndarray, team: int) -> float:
    target_x = GOAL_X if team == TeamID.RED else -GOAL_X
    target_y = float(np.clip(ball_pos[1], -GOAL_HALF_HEIGHT, GOAL_HALF_HEIGHT))
    return float(np.hypot(target_x - ball_pos[0], target_y - ball_pos[1]))


class ShapedReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self._prev = {}

    def _snapshot(self, player: PlayerHandler, state: GameState) -> dict:
        if player.team == TeamID.RED:
            my_score, opp_score = state.red_score, state.blue_score
        else:
            my_score, opp_score = state.blue_score, state.red_score
        return {
            "ball_dist": float(np.linalg.norm(state.ball.position - player.disc.position)),
            "ball_goal_dist": _ball_dist_to_goal(state.ball.position, player.team),
            "touches": player.player_data.number_touch,
            "my_score": my_score,
            "opp_score": opp_score,
        }

    def reset(self, initial_state: GameState, optional_data=None):
        self._prev = {}
        for player in initial_state.players:
            self._prev[player.id] = self._snapshot(player, initial_state)

    def get_reward(
        self,
        player: PlayerHandler,
        state: GameState,
        previous_action: np.ndarray,
        optional_data=None,
    ) -> float:
        prev = self._prev.get(player.id)
        cur = self._snapshot(player, state)
        self._prev[player.id] = cur
        if prev is None:
            return 0.0

        reward = STEP_PENALTY
        reward += (prev["ball_dist"] - cur["ball_dist"]) * APPROACH_WEIGHT * 50.0
        reward += (prev["ball_goal_dist"] - cur["ball_goal_dist"]) * BALL_TO_GOAL_WEIGHT * 50.0
        if cur["touches"] > prev["touches"]:
            reward += TOUCH_REWARD
        if cur["my_score"] > prev["my_score"]:
            reward += GOAL_REWARD
        if cur["opp_score"] > prev["opp_score"]:
            reward -= GOAL_REWARD
        return float(reward)

    def get_final_reward(
        self,
        player: PlayerHandler,
        state: GameState,
        previous_action: np.ndarray,
        optional_data=None,
    ) -> float:
        # Wywolywane raz, na ostatnim kroku epizodu. Jesli epizod skonczyl sie
        # bez gola (timeout), dokladamy jawna kare - stara wersja JS nie
        # oznaczala timeoutu jako konca i psulo to wartosci na koncowkach.
        reward = self.get_reward(player, state, previous_action, optional_data)
        cur = self._prev.get(player.id)
        if cur is not None and cur["my_score"] == cur["opp_score"]:
            reward += TIMEOUT_PENALTY
        return float(reward)
