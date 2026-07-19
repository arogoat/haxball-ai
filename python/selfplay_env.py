# Pojedyncze (z perspektywy SB3) srodowisko 1v1 z self-play:
#  - uczacy sie agent steruje czerwonym,
#  - niebieskim steruje zamrozony snapshot polityki losowany z puli (opponents/)
#    - 80% szans na najnowszy, 20% na losowy starszy ("liga" w miniaturze,
#    zapobiega przeuczaniu sie na jedna, aktualna wersje samego siebie),
#  - dopoki puli nie ma (poczatek treningu), przeciwnik gra losowo.
# Obserwacje sa lustrzane per-druzyna (obs.py), wiec ta sama siec gra poprawnie
# po obu stronach - dokladnie jak w wersji JS.
import os
import random
from typing import Optional

import gymnasium
import numpy as np
from gymnasium import spaces

OPPONENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opponents")
TICK_SKIP = 5  # decyzja co 5 tickow (~83ms) - jak ACTION_REPEAT w wersji JS
EPISODE_STEPS = 360  # 360 decyzji * 5 tickow = 1800 tickow = 30s meczu

_OBS_SIZE = 15


def _make_haxball_env():
    from ursinaxball import Game
    import haxballgym
    from haxballgym.utils.terminal_conditions.common_conditions import (
        TimeoutCondition,
        GoalScoredCondition,
    )

    from obs import NormalizedObs
    from rewards import ShapedReward
    from state_setters import RandomState

    game = Game(
        enable_renderer=False,
        enable_recorder=False,
        enable_vsync=False,
        logging_level=40,
    )
    return haxballgym.make(
        game=game,
        team_size=1,
        tick_skip=TICK_SKIP,
        reward_fn=ShapedReward(),
        obs_builder=NormalizedObs(),
        state_setter=RandomState(kickoff_prob=0.2),
        terminal_conditions=[TimeoutCondition(EPISODE_STEPS), GoalScoredCondition()],
    )


class _OpponentPool:
    """Trzyma zamrozona polityke przeciwnika, odswieza z dysku co epizod."""

    def __init__(self):
        self._policy = None
        self._loaded_path: Optional[str] = None

    def _pick_path(self) -> Optional[str]:
        if not os.path.isdir(OPPONENTS_DIR):
            return None
        snaps = sorted(f for f in os.listdir(OPPONENTS_DIR) if f.endswith(".zip"))
        if not snaps:
            return None
        if len(snaps) == 1 or random.random() < 0.8:
            return os.path.join(OPPONENTS_DIR, snaps[-1])
        return os.path.join(OPPONENTS_DIR, random.choice(snaps[:-1]))

    def refresh(self):
        path = self._pick_path()
        if path is None or path == self._loaded_path:
            return
        try:
            from stable_baselines3 import PPO
            self._policy = PPO.load(path, device="cpu")
            self._loaded_path = path
        except Exception:
            pass  # uszkodzony/pisany wlasnie plik - gramy dalej stara polityka

    def act(self, obs: np.ndarray, action_space) -> np.ndarray:
        if self._policy is None:
            return action_space.sample()
        action, _ = self._policy.predict(obs, deterministic=False)
        return action


class HaxballSelfPlayEnv(gymnasium.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self._env = _make_haxball_env()
        self._pool = _OpponentPool()
        self.action_space = spaces.MultiDiscrete([3, 3, 2])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(_OBS_SIZE,), dtype=np.float32
        )
        self._opponent_obs = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._pool.refresh()
        obs_all = self._env.reset()
        self._opponent_obs = np.asarray(obs_all[1], dtype=np.float32)
        return np.asarray(obs_all[0], dtype=np.float32), {}

    def step(self, action):
        opp_action = self._pool.act(self._opponent_obs, self.action_space)
        obs_all, reward_all, done, info = self._env.step(
            [np.asarray(action), np.asarray(opp_action)]
        )
        self._opponent_obs = np.asarray(obs_all[1], dtype=np.float32)
        reward = float(reward_all[0]) if isinstance(reward_all, (list, tuple, np.ndarray)) else float(reward_all)
        # stary interfejs gym (done) -> gymnasium (terminated/truncated);
        # rozroznienie nie ma tu znaczenia praktycznego, wybieramy terminated
        return (
            np.asarray(obs_all[0], dtype=np.float32),
            reward,
            bool(done),
            False,
            info if isinstance(info, dict) else {},
        )
