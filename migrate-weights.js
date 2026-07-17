// Jednorazowy skrypt: przenosi to, czego bot już się nauczył (dqn-weights.json,
// stary model "dojdź do piłki", 5 wejść) do nowego modelu z bramką (7 wejść).
//
// Jak to działa: warstwy ukryte (24->24) i wyjściowa (24->8) są identyczne w obu
// modelach (ta sama liczba neuronów) - kopiujemy je 1:1, bez zmian. Zmieniła się
// tylko warstwa wejściowa (5 wejść -> 7 wejść), bo doszły 2 nowe cechy (odległość
// piłki od bramki). Dla niej kopiujemy pierwsze 5 wierszy (to, czego bot już się
// nauczył), a 2 nowe wiersze (dla nowych cech) zostają z losowej inicjalizacji -
// ich się jeszcze nikt nie nauczył, więc nie ma czego kopiować.
const fs = require("fs");
const { createModel } = require("./dqn-common.js");

const OLD_PATH = "dqn-weights.json";
const NEW_PATH = "dqn-weights-goal.json";

if (!fs.existsSync(OLD_PATH)) {
  console.log(`Nie znaleziono ${OLD_PATH} w tym folderze - nie ma czego migrować.`);
  process.exit(1);
}

const oldWeights = JSON.parse(fs.readFileSync(OLD_PATH, "utf8"));
const [oldKernel0, oldBias0, oldKernel1, oldBias1, oldKernel2, oldBias2] = oldWeights;

if (oldKernel0.length !== 5) {
  console.log(`Uwaga: spodziewałem się starego modelu z 5 wejściami, a mam ${oldKernel0.length}. Sprawdź, czy to na pewno ten plik.`);
  process.exit(1);
}

const newModel = createModel();
const newWeights = newModel.getWeights().map((w) => w.arraySync());
const [newKernel0] = newWeights;

for (let i = 0; i < oldKernel0.length; i++) {
  newKernel0[i] = oldKernel0[i];
}
// wiersze 5 i 6 w newKernel0 (nowe cechy: piłka->bramka X, piłka->bramka Y)
// zostają nietknięte - to ich losowa inicjalizacja z createModel()

const migratedWeights = [
  newKernel0,
  oldBias0,
  oldKernel1,
  oldBias1,
  oldKernel2,
  oldBias2,
];

fs.writeFileSync(NEW_PATH, JSON.stringify(migratedWeights));
console.log(`Gotowe. Zapisano ${NEW_PATH}:`);
console.log(`  - warstwy ukryte i wyjściowa: 1:1 skopiowane ze starego modelu (to, czego się już nauczył)`);
console.log(`  - warstwa wejściowa: pierwsze 5 wierszy skopiowane, 2 nowe (bramka) losowe - te dopiero się nauczy`);
