# Srodowisko 2v2 self-play (futsal) z perspektywy SB3 jako pojedynczy agent.
# Uczaca sie polityka steruje OBIEMA czerwonymi (odpytywana OSOBNO dla kazdego,
# z egocentryczna obserwacja - patrz obs_2v2.py), a przeciwnik (zamrozony
# snapshot z puli) steruje OBIEMA niebieskimi.
#
# Kluczowy trik "shared policy": SB3 widzi to jako zwykle srodowisko 1-agentowe,
# ale kazdy krok obejmuje OBU czerwonych. Zeby oba przyklady (gracz 0 i gracz 1)
# trafialy do uczenia z pelnym sygnalem, na przemian raportujemy SB3 perspektywe
# raz jednego, raz drugiego czerwonego gracza (oba i tak steruje ta sama siec,
# wiec uczy sie z obu pozycji). Akcje dla obu czerwonych generuje ta sama biezaca
# polityka: dla "raportowanego" - akcja od SB3, dla drugiego - z wewnetrznej kopii.
import os
import random
from typing import Optional

import gymnasium
import numpy as np
from gymnasium import spaces

OPPONENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opponents_2v2")
TICK_SKIP = 5
EPISODE_STEPS = 400  # 400 decyzji * 5 tickow = 2000 tickow ~ 33s (futsal wolniejszy)

from obs_2v2 import OBS_SIZE  # noqa: E402


def _make_env():
    from ursinaxball import Game
    import haxballgym
    from haxballgym.utils.terminal_conditions.common_conditions import TimeoutCondition

    from obs_2v2 import Ego2v2Obs
    from rewards_2v2 import TeamReward2v2
    from state_setters_futsal import FutsalRandomState
    from terminal_conditions import FixedGoalScoredCondition
    from futsal_config import STADIUM_FILE

    game = Game(
        stadium_file=STADIUM_FILE,
        enable_renderer=False,
        enable_recorder=False,
        enable_vsync=False,
        logging_level=40,
    )
    return haxballgym.make(
        game=game,
        team_size=2,
        tick_skip=TICK_SKIP,
        reward_fn=TeamReward2v2(),
        obs_builder=Ego2v2Obs(),
        state_setter=FutsalRandomState(kickoff_prob=0.2),
        terminal_conditions=[TimeoutCondition(EPISODE_STEPS), FixedGoalScoredCondition()],
    )


class _SnapshotPolicy:
    """Laduje polityke ze snapshotu na dysku. Uzywana i dla przeciwnika
    (niebiescy, losowy z puli), i dla partnera (czerwony, zawsze najnowszy)."""

    def __init__(self, always_latest: bool):
        self._always_latest = always_latest
        self._policy = None
        self._loaded = None

    def _pick(self) -> Optional[str]:
        if not os.path.isdir(OPPONENTS_DIR):
            return None
        snaps = sorted(f for f in os.listdir(OPPONENTS_DIR) if f.endswith(".zip"))
        if not snaps:
            return None
        if self._always_latest or len(snaps) == 1 or random.random() < 0.7:
            return os.path.join(OPPONENTS_DIR, snaps[-1])
        return os.path.join(OPPONENTS_DIR, random.choice(snaps[:-1]))

    def refresh(self):
        p = self._pick()
        if p is None or p == self._loaded:
            return
        try:
            from stable_baselines3 import PPO
            self._policy = PPO.load(p, device="cpu")
            self._loaded = p
        except Exception:
            pass

    def act(self, obs, action_space):
        if self._policy is None:
            return action_space.sample()
        a, _ = self._policy.predict(obs, deterministic=False)
        return a


class Haxball2v2SelfPlayEnv(gymnasium.Env):
    metadata = {"render_modes": []}

    # indeksy graczy: 0,1 = czerwoni (uczeni), 2,3 = niebiescy (przeciwnik)
    RED = [0, 1]
    BLUE = [2, 3]

    def __init__(self):
        super().__init__()
        self._env = _make_env()
        self._pool = _SnapshotPolicy(always_latest=False)      # przeciwnik: losowy z puli
        self._partner = _SnapshotPolicy(always_latest=True)     # partner: zawsze najnowszy
        self.action_space = spaces.MultiDiscrete([3, 3, 2])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32)
        self._obs_all = None
        self._reported = 0  # ktory czerwony (0 lub 1) jest raportowany do SB3 w tym epizodzie

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._pool.refresh()
        self._partner.refresh()
        self._obs_all = self._env.reset()
        self._reported = random.randint(0, 1)  # na zmiane uczymy sie z obu pozycji
        red_idx = self.RED[self._reported]
        return np.asarray(self._obs_all[red_idx], dtype=np.float32), {}

    def step(self, action):
        reported_red = self.RED[self._reported]
        partner_red = self.RED[1 - self._reported]

        actions = [None, None, None, None]
        actions[reported_red] = np.asarray(action)
        # partner (drugi czerwony): najnowszy snapshot = niedawna wersja siebie
        actions[partner_red] = np.asarray(self._partner.act(np.asarray(self._obs_all[partner_red], dtype=np.float32), self.action_space))
        # niebiescy: przeciwnik losowany z puli
        for b in self.BLUE:
            actions[b] = np.asarray(self._pool.act(np.asarray(self._obs_all[b], dtype=np.float32), self.action_space))

        obs_all, reward_all, done, info = self._env.step(actions)
        self._obs_all = obs_all
        reward = float(reward_all[reported_red]) if isinstance(reward_all, (list, tuple, np.ndarray)) else float(reward_all)
        return (
            np.asarray(obs_all[reported_red], dtype=np.float32),
            reward,
            bool(done),
            False,
            info if isinstance(info, dict) else {},
        )
