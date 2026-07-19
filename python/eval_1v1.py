# Ewaluacja: mecze checkpoint vs checkpoint (albo vs losowy przeciwnik),
# z NAGRANIAMI do obejrzenia w przegladarce.
#   xvfb-run -a python3 eval_1v1.py                       # latest vs latest, 20 meczow
#   xvfb-run -a python3 eval_1v1.py --opponent random     # latest vs losowy
#   xvfb-run -a python3 eval_1v1.py --games 50 --no-rec   # bez nagran
# Nagrania laduja w recordings/ - otworz https://wazarr94.github.io/ i wczytaj
# plik nagrania, zeby OBEJRZEC mecz (bez tokenow, bez node-haxball!).
import argparse
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LATEST_PATH = os.path.join(BASE_DIR, "checkpoints", "latest.zip")
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")

from selfplay_env import TICK_SKIP, EPISODE_STEPS  # noqa: E402


def make_eval_env(record: bool):
    from ursinaxball import Game
    import haxballgym
    from haxballgym.utils.terminal_conditions.common_conditions import (
        TimeoutCondition,
        GoalScoredCondition,
    )
    from obs import NormalizedObs
    from rewards import ShapedReward
    from state_setters import RandomState

    if record:
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
    game = Game(
        enable_renderer=False,
        enable_recorder=record,
        enable_vsync=False,
        logging_level=40,
        folder_rec=RECORDINGS_DIR + os.sep if record else "",
    )
    return haxballgym.make(
        game=game,
        team_size=1,
        tick_skip=TICK_SKIP,
        reward_fn=ShapedReward(),
        obs_builder=NormalizedObs(),
        # te same losowe starty co w treningu - mecze eval sa dzieki temu
        # rozne (deterministyczna polityka + staly start = 10x identyczny mecz)
        state_setter=RandomState(kickoff_prob=0.2),
        terminal_conditions=[TimeoutCondition(EPISODE_STEPS), GoalScoredCondition()],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=LATEST_PATH)
    parser.add_argument("--opponent", default=None, help='sciezka do zip albo "random"')
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--no-rec", action="store_true")
    parser.add_argument("--stochastic", action="store_true",
                        help="akcje losowane z rozkladu polityki (jak w treningu) zamiast deterministycznych - dwie identyczne deterministyczne sieci potrafia wpasc w patowa petle")
    args = parser.parse_args()
    det = not args.stochastic

    from stable_baselines3 import PPO

    model = PPO.load(args.model, device="cpu")
    opponent = None
    if args.opponent is None:
        opponent = model
    elif args.opponent != "random":
        opponent = PPO.load(args.opponent, device="cpu")

    env = make_eval_env(record=not args.no_rec)

    record = not args.no_rec
    red_wins = blue_wins = draws = 0
    lengths = []
    for game_i in range(args.games):
        # save_recording=True zapisuje nagranie POPRZEDNIEGO epizodu (tak dziala
        # haxballgym) - przy pierwszym reset nie ma czego zapisywac
        obs_all = env.reset(save_recording=record and game_i > 0)
        done = False
        steps = 0
        info = {}
        while not done:
            red_action, _ = model.predict(np.asarray(obs_all[0], dtype=np.float32), deterministic=det)
            if opponent is None:
                blue_action = env.action_space.sample()
            else:
                blue_action, _ = opponent.predict(np.asarray(obs_all[1], dtype=np.float32), deterministic=det)
            obs_all, _, done, info = env.step([np.asarray(red_action), np.asarray(blue_action)])
            steps += 1
        lengths.append(steps)

        state = info.get("state") if isinstance(info, dict) else None
        red_s = getattr(state, "red_score", None)
        blue_s = getattr(state, "blue_score", None)
        if red_s is None:
            print(f"mecz {game_i + 1}: koniec po {steps} krokach (brak wyniku w info)")
            continue
        if red_s > blue_s:
            red_wins += 1
            result = "czerwoni"
        elif blue_s > red_s:
            blue_wins += 1
            result = "niebiescy"
        else:
            draws += 1
            result = "remis"
        print(f"mecz {game_i + 1}: {result} ({red_s}:{blue_s}, {steps} krokow)")

    if record:
        env.reset(save_recording=True)  # zapis nagrania ostatniego meczu

    n = args.games
    print()
    print(f"Wyniki po {n} meczach: czerwoni {red_wins} ({100*red_wins/n:.0f}%), "
          f"niebiescy {blue_wins} ({100*blue_wins/n:.0f}%), remisy {draws} ({100*draws/n:.0f}%), "
          f"sr. dlugosc {np.mean(lengths):.0f} krokow")
    if not args.no_rec:
        print(f"Nagrania: {RECORDINGS_DIR} -> otworz https://wazarr94.github.io/ i wczytaj plik")


if __name__ == "__main__":
    main()
