#!/bin/bash
# Startuje trening 2v2 w tmux (petla auto-restartu + log). Bezpieczny do
# wielokrotnego uruchamiania. Podpiety pod cron @reboot (patrz README).
cd "$(dirname "$0")" || exit 1

if tmux has-session -t trening-2v2 2>/dev/null; then
  echo "Sesja trening-2v2 juz dziala - nic nie robie."
  exit 0
fi

tmux new -d -s trening-2v2 'while true; do xvfb-run -a ./venv/bin/python train_2v2.py 2>&1 | tee -a train_2v2.log; echo "=== RESTART $(date) ===" | tee -a train_2v2.log; sleep 5; done'
echo "$(date '+%Y-%m-%d %H:%M:%S') - wystartowano sesje trening-2v2"
