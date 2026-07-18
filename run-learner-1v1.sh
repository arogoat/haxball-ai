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
  if [ "$code" -eq 0 ]; then
    # Kod 0 = learner sam sie zakonczyl PO OSIAGNIECIU TOTAL_EPISODES (celowo,
    # patrz proces.exit(0) w learner-1v1.js). Bez tego rozroznienia petla
    # restartowala go w kolko, a kazdy restart odpalal sie na chwile, widzial
    # ze limit juz przekroczony i od razu konczyl - epizody rosly w nieskonczonosc
    # (30007 -> 30141 -> ...) mimo "zakonczonego" treningu.
    echo "$(date '+%Y-%m-%d %H:%M:%S') - learner-1v1.js zakonczony poprawnie (limit epizodow osiagniety) - nie restartuje"
    break
  fi
  echo "$(date '+%Y-%m-%d %H:%M:%S') - learner-1v1.js padl (kod $code), restart za 5s"
  sleep 5
done
