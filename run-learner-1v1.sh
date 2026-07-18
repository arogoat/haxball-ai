#!/bin/bash
# Uruchamiany NA SERWERZE zamiast bezposredniego "node learner-1v1.js".
# Odpala learnera w petli - jesli caly proces padnie (blad spoza obslugi w
# kodzie, OOM, zerwane polaczenie z node-haxball itp.), startuje go ponownie
# automatycznie zamiast zostawiac trening martwy na reszte nocy.
#
# Wagi i postep sa wczytywane z plikow (dqn-weights-1v1.json,
# training-progress-1v1.json) przy kazdym starcie, wiec restart nie traci
# dotychczasowego treningu - kontynuuje od ostatniego zapisu.
cd "$(dirname "$0")" || exit 1

while true; do
  echo "$(date '+%Y-%m-%d %H:%M:%S') - startuje learner-1v1.js"
  node learner-1v1.js
  code=$?
  echo "$(date '+%Y-%m-%d %H:%M:%S') - learner-1v1.js zakonczony (kod $code), restart za 5s"
  sleep 5
done
