# HaxBall AI - trening w Pythonie (PPO + self-play)

Nowa generacja treningu botow: zamiast recznego DQN w JS, uzywamy gotowej
fizyki haxballa w Pythonie ([haxballgym](https://github.com/HaxballGym/HaxballGym) /
[ursinaxball](https://github.com/HaxballGym/Ursinaxball)) i przemyslowej
implementacji PPO (stable-baselines3). GPU NIE jest potrzebne.

## Instalacja na serwerze (Ubuntu, jednorazowo)

```bash
sudo apt-get install -y xvfb python3-venv
cd ~/haxball-ai/python
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

Uwaga: `xvfb` jest potrzebny, bo silnik (ursina) wymaga "ekranu" nawet w trybie
headless - `xvfb-run` tworzy wirtualny. Wymagany Python >= 3.10.

## Trening

```bash
cd ~/haxball-ai/python
tmux new -d -s trening-py "xvfb-run -a ./venv/bin/python train_1v1.py"
tmux capture-pane -t trening-py -p | tail -20   # podglad logow
```

- Wznawia automatycznie z `checkpoints/latest.zip` - mozna bezpiecznie
  restartowac.
- Co 100k krokow: checkpoint + snapshot przeciwnika do puli self-play
  (`opponents/`), stare snapshoty rotowane (max 20).
- Metryki: `ep_rew_mean` w logu konsoli powinno rosnac. Pelne wykresy:
  `tensorboard --logdir logs/`.

## Ogladanie meczow (bez tokenow!)

```bash
xvfb-run -a ./venv/bin/python eval_1v1.py --games 10
```

Nagrania meczow laduja w `recordings/`. Otworz https://wazarr94.github.io/
w przegladarce i wczytaj plik nagrania - zobaczysz caly mecz jak w haxballu.

Warianty: `--opponent random` (vs losowy), `--opponent sciezka/do/opp_xxx.zip`
(vs starszy snapshot - pokazuje postep), `--no-rec` (bez nagran, szybciej).

## Struktura

- `obs.py` - znormalizowane obserwacje (lustrzane per druzyna -> jeden model
  gra po obu stronach)
- `rewards.py` - shaped reward (port naprawionej wersji z JS, skala PPO)
- `selfplay_env.py` - srodowisko gymnasium: agent=czerwony, przeciwnik=zamrozony
  snapshot z puli (80% najnowszy / 20% losowy starszy)
- `train_1v1.py` - PPO, 4 rownolegle srodowiska, auto-wznawianie
- `eval_1v1.py` - mecze + nagrania do przegladarki

## Dalsze kroki (po udanym 1v1)

1. Wieksza siec / dluzszy trening, jesli 1v1 dziala ale plateau.
2. `team_size=2` (2v2) - haxballgym wspiera to natywnie; wymaga rozszerzenia
   obserwacji o kolege z druzyny.
3. Docelowo 5v5 + curriculum (1v1 -> 2v2 -> 3v3 -> 5v5).
4. Pomost do prawdziwych pokojow (bot-klient node-haxball sterowany polityka
   z Pythona) - do meczow z ludzmi.
