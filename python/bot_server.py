# Serwer polityki bota (1v1 classic) - mostek do grania z botem w prawdziwym
# haxballu. Nasluchuje na lokalnym TCP; dostaje SUROWY stan gry od node
# (play-vs-bot.js), buduje wektor obserwacji DOKLADNIE tak jak trening
# (te same stale normalizacji co obs.py) i zwraca akcje sieci [a0,a1,a2].
#
# Node przysyla surowe liczby (nie gotowa obserwacje), zeby normalizacja zyla
# tylko tu, w Pythonie - jedno zrodlo prawdy, brak ryzyka rozjazdu miedzy JS a PY.
#
# Uruchamianie:
#   ./venv/bin/python bot_server.py [--model checkpoints/latest.zip] [--port 5555]
import argparse
import json
import os
import socket

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# te same stale co obs.py (classic 1v1)
POS_SCALE = 400.0
VEL_SCALE = 5.0


def build_obs(s: dict, mirror_x: float) -> np.ndarray:
    """Buduje 15-wymiarowa obserwacje z surowego stanu. mirror_x = -1 dla
    niebieskiego (lustro), 1 dla czerwonego. s zawiera pozycje/predkosci pilki,
    'ja' i przeciwnika oraz prev (poprzednia akcja [a0,a1,a2])."""
    mx = np.array([mirror_x, 1.0])
    obs = []
    obs += list(np.array([s["ballX"], s["ballY"]]) * mx / POS_SCALE)
    obs += list(np.array([s["ballVX"], s["ballVY"]]) * mx / VEL_SCALE)
    obs += list(np.asarray(s.get("prev", [0, 0, 0]), dtype=float))
    obs += list(np.array([s["selfX"], s["selfY"]]) * mx / POS_SCALE)
    obs += list(np.array([s["selfVX"], s["selfVY"]]) * mx / VEL_SCALE)
    obs += list(np.array([s["oppX"], s["oppY"]]) * mx / POS_SCALE)
    obs += list(np.array([s["oppVX"], s["oppVY"]]) * mx / VEL_SCALE)
    return np.asarray(obs, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(BASE_DIR, "checkpoints", "latest.zip"))
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--deterministic", action="store_true",
                        help="bez losowosci (moze wpasc w pat); domyslnie lekko stochastyczny - grywalniej")
    args = parser.parse_args()

    from stable_baselines3 import PPO
    print(f"Laduje model: {args.model}")
    model = PPO.load(args.model, device="cpu")
    det = args.deterministic

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port))
    srv.listen(1)
    print(f"Serwer bota gotowy na porcie {args.port}. Czekam na node...")

    while True:
        conn, _ = srv.accept()
        print("node polaczony")
        buf = b""
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    s = json.loads(line)
                    mirror_x = -1.0 if s.get("team") == 2 else 1.0
                    obs = build_obs(s, mirror_x)
                    action, _ = model.predict(obs, deterministic=det)
                    conn.sendall((json.dumps([int(a) for a in action]) + "\n").encode())
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            conn.close()
            print("node rozlaczony, czekam ponownie...")


if __name__ == "__main__":
    main()
