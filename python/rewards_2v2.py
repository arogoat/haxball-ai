# Reward druzynowy 2v2 (futsal). Rdzen jak w 1v1 (shaping: zblizanie do pilki +
# pchanie pilki w strone bramki, bonus za dotkniecie, gol +/-, kara za timeout),
# ale wspolrzedne bramek z mapy futsalowej i DWA komponenty:
#   - indywidualny: zeby kazdy gracz mial gesty sygnal (nie tylko rzadki gol)
#   - druzynowy: gol liczy sie dla OBU graczy druzyny (uczy gry na zespol, nie
#     egoistycznego zbierania shapingu)
import numpy as np
from ursinaxball.common_values import TeamID
from ursinaxball.modules import PlayerHandler

from haxballgym.utils.gamestates import GameState
from haxballgym.utils.reward_functions import RewardFunction

from futsal_config import GOAL_X, GOAL_HALF_HEIGHT

# skale dobrane do wiekszego boiska (dystanse ~3x wieksze niz classic)
STEP_PENALTY = -0.01
APPROACH_WEIGHT = 0.1 / 150.0
BALL_TO_GOAL_WEIGHT = 0.2 / 150.0
TOUCH_REWARD = 0.3
GOAL_REWARD = 10.0
TIMEOUT_PENALTY = -4.0


def _ball_dist_to_goal(ball_pos: np.ndarray, team: int) -> float:
    target_x = GOAL_X if team == TeamID.RED else -GOAL_X
    target_y = float(np.clip(ball_pos[1], -GOAL_HALF_HEIGHT, GOAL_HALF_HEIGHT))
    return float(np.hypot(target_x - ball_pos[0], target_y - ball_pos[1]))


class TeamReward2v2(RewardFunction):
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

    def get_reward(self, player, state, previous_action, optional_data=None) -> float:
        prev = self._prev.get(player.id)
        cur = self._snapshot(player, state)
        self._prev[player.id] = cur
        if prev is None:
            return 0.0

        reward = STEP_PENALTY
        # indywidualny shaping (gesty sygnal)
        reward += (prev["ball_dist"] - cur["ball_dist"]) * APPROACH_WEIGHT * 150.0
        reward += (prev["ball_goal_dist"] - cur["ball_goal_dist"]) * BALL_TO_GOAL_WEIGHT * 150.0
        if cur["touches"] > prev["touches"]:
            reward += TOUCH_REWARD
        # druzynowy: gol liczy sie dla obu graczy druzyny (my_score/opp_score sa
        # wspolne dla druzyny, wiec kazdy gracz dostaje ten sam sygnal bramkowy)
        if cur["my_score"] > prev["my_score"]:
            reward += GOAL_REWARD
        if cur["opp_score"] > prev["opp_score"]:
            reward -= GOAL_REWARD
        return float(reward)

    def get_final_reward(self, player, state, previous_action, optional_data=None) -> float:
        reward = self.get_reward(player, state, previous_action, optional_data)
        cur = self._prev.get(player.id)
        if cur is not None and cur["my_score"] == cur["opp_score"]:
            reward += TIMEOUT_PENALTY
        return float(reward)
