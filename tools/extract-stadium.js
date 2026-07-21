// Wyciaga pelna definicje mapy z nagrania .hbr2 i zapisuje w formacie
// zjadliwym dla ursinaxball (symulator treningowy w Pythonie).
//
// Uzycie:
//   node tools/extract-stadium.js nagranie.hbr2 [wyjscie.json5]
//
// Roznice formatow, ktore naprawiamy:
//  - haxball: pilka to discs[0] + ballPhysics="disc0"; ursinaxball chce
//    ballPhysics jako obiekt (i sam wstawia go jako discs[0])
//  - ursinaxball wymaga pola "traits" (choc pustego)
// Przetestowane end-to-end: mapa Futsal x7 z nagrania dziala w symulatorze.
const fs = require("fs");

const inputPath = process.argv[2];
if (!inputPath) {
  console.error("Uzycie: node tools/extract-stadium.js <nagranie.hbr2> [wyjscie.json5]");
  process.exit(1);
}

const { Replay, Utils } = require("node-haxball")();
const data = fs.readFileSync(inputPath, null);
const rd = Replay.readAll(data);
const stadium = rd.roomData?.stadium;
if (!stadium) {
  console.error("Brak stadionu w nagraniu (uszkodzony plik?)");
  process.exit(1);
}

const exported = Utils.exportStadium(stadium);
const obj = typeof exported === "string" ? JSON.parse(exported) : exported;

// konwersja do formatu ursinaxball
if (obj.ballPhysics === "disc0" && Array.isArray(obj.discs) && obj.discs.length > 0) {
  const ball = obj.discs.shift();
  delete ball.pos;
  delete ball.speed;
  obj.ballPhysics = ball;
}
if (!obj.traits) obj.traits = {};

const safeName = (stadium.name || "mapa").replace(/[^a-zA-Z0-9 _.-]/g, "").trim().replace(/\s+/g, "-").slice(0, 60);
const outPath = process.argv[3] || `python/stadiums/${safeName}.json5`;
fs.mkdirSync(require("path").dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, JSON.stringify(obj, null, 1));
console.log(`Stadion "${stadium.name}" zapisany do ${outPath}`);
console.log(`(vertexes: ${obj.vertexes?.length}, segments: ${obj.segments?.length}, discs: ${obj.discs?.length}+pilka, goals: ${obj.goals?.length})`);
