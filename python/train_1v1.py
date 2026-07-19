# Trening PPO 1v1 z self-play.
# Uruchamianie (na serwerze, w venv, patrz README):
#   xvfb-run -a python3 train_1v1.py
# Wznawia automatycznie z checkpoints/latest.zip jesli istnieje.
# Postep: tensorboard --logdir logs/  (albo obserwuj konsole)
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from selfplay_env import HaxballSelfPlayEnv, OPPONENTS_DIR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
LATEST_PATH = os.path.join(CHECKPOINTS_DIR, "latest.zip")

N_ENVS = 4
TOTAL_TIMESTEPS = 20_000_000
SNAPSHOT_EVERY = 100_000  # co tyle krokow: nowy przeciwnik w puli + checkpoint
MAX_OPPONENTS = 20  # najstarsze snapshoty ponad limit sa usuwane


class SelfPlayCallback(BaseCallback):
    def _on_step(self) -> bool:
        if self.num_timesteps > 0 and self.num_timesteps % SNAPSHOT_EVERY < self.training_env.num_envs:
            os.makedirs(OPPONENTS_DIR, exist_ok=True)
            os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
            snap_path = os.path.join(OPPONENTS_DIR, f"opp_{self.num_timesteps:010d}.zip")
            if not os.path.exists(snap_path):
                self.model.save(snap_path)
                self.model.save(LATEST_PATH)
                if self.verbose:
                    print(f"[self-play] snapshot + checkpoint @ {self.num_timesteps} krokow")
                snaps = sorted(f for f in os.listdir(OPPONENTS_DIR) if f.endswith(".zip"))
                for old in snaps[:-MAX_OPPONENTS]:
                    os.remove(os.path.join(OPPONENTS_DIR, old))
        return True


def make_env():
    def _init():
        return Monitor(HaxballSelfPlayEnv())
    return _init


def main():
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])

    if os.path.exists(LATEST_PATH):
        print(f"Wznawiam trening z {LATEST_PATH}")
        model = PPO.load(LATEST_PATH, env=env, device="cpu", tensorboard_log=LOGS_DIR)
    else:
        print("Start od zera (brak checkpointu)")
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
