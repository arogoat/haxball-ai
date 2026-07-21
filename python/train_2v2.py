# Trening PPO 2v2 (futsal) z self-play i wspoldzielona polityka wieloagentowa.
# Uruchamianie:
#   xvfb-run -a python3 train_2v2.py
# Wznawia z checkpoints_2v2/latest.zip. Snapshoty przeciwnika/partnera do
# opponents_2v2/. Wieksza siec niz w 1v1 - gra druzynowa wymaga wiecej pojemnosci.
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from selfplay_env_2v2 import Haxball2v2SelfPlayEnv, OPPONENTS_DIR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints_2v2")
LOGS_DIR = os.path.join(BASE_DIR, "logs_2v2")
LATEST_PATH = os.path.join(CHECKPOINTS_DIR, "latest.zip")

N_ENVS = 4
TOTAL_TIMESTEPS = 100_000_000
SNAPSHOT_EVERY = 200_000  # co tyle: nowy snapshot (przeciwnik+partner) + checkpoint
MAX_OPPONENTS = 40


class SelfPlayCallback(BaseCallback):
    def _on_step(self) -> bool:
        if self.num_timesteps > 0 and self.num_timesteps % SNAPSHOT_EVERY < self.training_env.num_envs:
            os.makedirs(OPPONENTS_DIR, exist_ok=True)
            os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
            snap = os.path.join(OPPONENTS_DIR, f"opp_{self.num_timesteps:012d}.zip")
            if not os.path.exists(snap):
                self.model.save(snap)
                self.model.save(LATEST_PATH)
                if self.verbose:
                    print(f"[2v2] snapshot + checkpoint @ {self.num_timesteps}")
                snaps = sorted(f for f in os.listdir(OPPONENTS_DIR) if f.endswith(".zip"))
                for old in snaps[:-MAX_OPPONENTS]:
                    os.remove(os.path.join(OPPONENTS_DIR, old))
        return True


def make_env():
    def _init():
        return Monitor(Haxball2v2SelfPlayEnv())
    return _init


def main():
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])

    if os.path.exists(LATEST_PATH):
        print(f"Wznawiam z {LATEST_PATH}")
        model = PPO.load(LATEST_PATH, env=env, device="cpu", tensorboard_log=LOGS_DIR)
    else:
        print("Start od zera (2v2 futsal)")
        model = PPO(
            "MlpPolicy",
            env,
            device="cpu",
            verbose=1,
            tensorboard_log=LOGS_DIR,
            n_steps=512,
            batch_size=512,
            learning_rate=3e-4,
            gamma=0.99,
            ent_coef=0.01,
            # wieksza siec dla gry druzynowej (domyslna to 64x64)
            policy_kwargs=dict(net_arch=[256, 256]),
        )

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=SelfPlayCallback(verbose=1),
            reset_num_timesteps=False,
            progress_bar=False,
        )
    finally:
        model.save(LATEST_PATH)
        print(f"Zapisano {LATEST_PATH}")


if __name__ == "__main__":
    main()
