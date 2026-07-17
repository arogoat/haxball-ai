#!/bin/bash
# Uruchamiany cyklicznie (cron) NA SERWERZE - commituje i wypycha na GitHub
# tylko pliki z wynikami treningu (wagi, postęp), nie ruszając kodu.
# Dzięki temu Claude (i Ty, lokalnie) może zobaczyć świeże wyniki bez czekania
# aż ktoś ręcznie pociągnie zmiany z serwera.
cd "$(dirname "$0")" || exit 1

git add dqn-weights-1v1.json training-progress-1v1.json dqn-weights-goal.json training-progress.json 2>/dev/null

if ! git diff --cached --quiet; then
  git commit -m "Auto-sync wynikow treningu $(date '+%Y-%m-%d %H:%M')" --quiet
  git push --quiet
  echo "$(date '+%Y-%m-%d %H:%M:%S') - wypchnieto zmiany"
else
  echo "$(date '+%Y-%m-%d %H:%M:%S') - brak zmian, pomijam"
fi
