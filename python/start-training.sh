#!/bin/bash
# Startuje trening PPO w tmux (z petla auto-restartu i logiem do train.log).
# Bezpieczny do wielokrotnego uruchamiania - jesli sesja juz dziala, nic nie robi.
# Podpiety pod cron @reboot, zeby nocne restarty serwera (auto-aktualizacje
# Ubuntu) nie zatrzymywaly treningu na wiele godzin.
cd "$(dirname "$0")" || exit 1

if tmux has-session -t trening-py 2>/dev/null; then
  echo "Sesja trening-py juz dziala - nic nie robie."
  exit 0
fi

tmux new -d -s trening-py 'while true; do xvfb-run -a ./venv/bin/python train_1v1.py 2>&1 | tee -a train.log; echo "=== RESTART $(date) ===" | tee -a train.log; sleep 5; done'
echo "$(date '+%Y-%m-%d %H:%M:%S') - wystartowano sesje trening-py"
