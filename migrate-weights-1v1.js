// Jednorazowy skrypt: przenosi to, czego nauczył się pojedynczy bot (dqn-weights-goal.json,
// 9 wejść: dojście do piłki + celowanie w bramkę) do UNIWERSALNEGO modelu 1v1 (14 wejść:
// to samo + świadomość przeciwnika). Warstwy ukryte/wyjściowa (24 neurony) są identyczne,
// więc kopiujemy je 1:1. Warstwa wejściowa: pierwsze 9 wierszy skopiowane, 5 nowych
// (przeciwnik) zostaje losowych.
//
// Jeden plik wyjściowy (dqn-weights-1v1.json) - ten sam model gra po obu stronach boiska.
const fs = require("fs");
const { createModel } = require("./dqn-common-1v1.js");

const OLD_PATH = "dqn-weights-goal.json";
const NEW_PATH = "dqn-weights-1v1.json";

if (!fs.existsSync(OLD_PATH)) {
  console.log(`Nie znaleziono ${OLD_PATH} - nie ma czego migrować.`);
  process.exit(1);
}

const oldWeights = JSON.parse(fs.readFileSync(OLD_PATH, "utf8"));
const [oldKernel0, oldBias0, oldKernel1, oldBias1, oldKernel2, oldBias2] = oldWeights;

const newModel = createModel();
const newWeights = newModel.getWeights().map((w) => w.arraySync());
const [newKernel0] = newWeights;

for (let i = 0; i < oldKernel0.length; i++) {
  newKernel0[i] = oldKernel0[i];
}
// wiersze 9-13 (5 nowych cech: pozycja/prędkość/dystans przeciwnika) zostają
// z losowej inicjalizacji createModel() - tego jeszcze nikt się nie nauczył

const migratedWeights = [newKernel0, oldBias0, oldKernel1, oldBias1, oldKernel2, oldBias2];
fs.writeFileSync(NEW_PATH, JSON.stringify(migratedWeights));
console.log(`Zapisano ${NEW_PATH} - warstwy ukryte/wyjściowa skopiowane, 5 nowych wejść (przeciwnik) losowe.`);
