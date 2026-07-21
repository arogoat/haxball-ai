# Ewaluacja 2v2 (futsal): czerwoni (nasz model, oba sterowane ta sama siecia)
# vs niebiescy (ten sam model albo snapshot/losowy). Z nagraniami do obejrzenia.
#   xvfb-run -a python3 eval_2v2.py --games 10 --kickoff   # wierne nagrania
#   xvfb-run -a python3 eval_2v2.py --games 20 --no-rec     # tylko statystyki
import argparse
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LATEST_PATH = os.path.join(BASE_DIR, "checkpoints_2v2", "latest.zip")
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")

from selfplay_env_2v2 import TICK_SKIP, EPISODE_STEPS  # noqa: E402


def make_env(record: bool, kickoff_only: bool):
    from ursinaxball import Game
    import haxballgym
    from haxballgym.utils.terminal_conditions.common_conditions import TimeoutCondition
    from obs_2v2 import Ego2v2Obs
    from rewards_2v2 import TeamReward2v2
    from state_setters_futsal import FutsalRandomState
    from terminal_conditions import FixedGoalScoredCondition
    from futsal_config import STADIUM_FILE

    if record:
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
    game = Game(
        stadium_file=STADIUM_FILE,
        enable_renderer=False,
        enable_recorder=record,
        enable_vsync=False,
        logging_level=40,
        folder_rec=RECORDINGS_DIR + os.sep if record else "",
    )
    return haxballgym.make(
        game=game,
        team_size=2,
        tick_skip=TICK_SKIP,
        reward_fn=TeamReward2v2(),
        obs_builder=Ego2v2Obs(),
        state_setter=FutsalRandomState(kickoff_prob=1.0 if kickoff_only else 0.2),
        terminal_conditions=[TimeoutCondition(EPISODE_STEPS), FixedGoalScoredCondition()],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=LATEST_PATH)
    parser.add_argument("--opponent", default=None, help='sciezka do zip, "random", albo brak = ten sam model')
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--no-rec", action="store_true")
    parser.add_argument("--kickoff", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()
    det = not args.stochastic

    from stable_baselines3 import PPO
    model = PPO.load(args.model, device="cpu")
    opponent = None
    if args.opponent is None:
        opponent = model
    elif args.opponent != "random":
        opponent = PPO.load(args.opponent, device="cpu")

    env = make_env(record=not args.no_rec, kickoff_only=args.kickoff)
    RED, BLUE = [0, 1], [2, 3]

    red_w = blue_w = draws = 0
    lengths = []
    for gi in range(args.games):
        obs_all = env.reset()
        done = False
        steps = 0
        info = {}
        while not done:
            actions = [None, None, None, None]
            for r in RED:
                a, _ = model.predict(np.asarray(obs_all[r], dtype=np.float32), deterministic=det)
                actions[r] = np.asarray(a)
            for b in BLUE:
                if opponent is None:
                    actions[b] = env.action_space.sample()
                else:
                    a, _ = opponent.predict(np.asarray(obs_all[b], dtype=np.float32), deterministic=det)
                    actions[b] = np.asarray(a)
            obs_all, _, done, info = env.step(actions)
            steps += 1
        lengths.append(steps)
        state = info.get("state") if isinstance(info, dict) else None
        rs, bs = getattr(state, "red_score", None), getattr(state, "blue_score", None)
        if rs is None:
            print(f"mecz {gi+1}: koniec po {steps} krokach")
            continue
        if rs > bs:
            red_w += 1; res = "czerwoni"
        elif bs > rs:
            blue_w += 1; res = "niebiescy"
        else:
            draws += 1; res = "remis"
        print(f"mecz {gi+1}: {res} ({rs}:{bs}, {steps} krokow)")

    n = args.games
    print(f"\nWyniki po {n}: czerwoni {red_w} ({100*red_w/n:.0f}%), niebiescy {blue_w} ({100*blue_w/n:.0f}%), "
          f"remisy {draws} ({100*draws/n:.0f}%), sr. dlugosc {np.mean(lengths):.0f}")
    if not args.no_rec:
        print(f"Nagrania: {RECORDINGS_DIR} -> https://wazarr94.github.io/")


if __name__ == "__main__":
    main()
