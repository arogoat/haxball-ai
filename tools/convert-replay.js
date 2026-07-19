// Konwerter nagran haxballa (.hbr2) -> dane treningowe (JSONL, 1 linia = 1 tick).
// Fundament pod imitation learning (nauka z gier dobrych graczy).
//
// Uzycie (na serwerze, w katalogu glownym repo - potrzebuje node_modules):
//   node tools/convert-replay.js sciezka/do/nagrania.hbr2 [wyjscie.jsonl]
//   node tools/convert-replay.js sciezka/do/nagrania.hbr2 --inspect   # podglad struktury stanu
//
// Kazda linia wyjscia: { f: nrKlatki, score: [red, blue], ball: {x,y,vx,vy},
//   players: [{ id, team, x, y, vx, vy, input }] }
// input = bitmaska klawiszy haxballa (1=gora, 2=dol, 4=lewo, 8=prawo, 16=kop).
//
// UWAGA: pisane wg dokumentacji node-haxball, ale dokladne ksztalty obiektow
// stanu moga sie roznic miedzy wersjami - przy pierwszym uruchomieniu na
// prawdziwym pliku uzyj --inspect i w razie potrzeby dopasujemy akcesory.
const fs = require("fs");

const inputPath = process.argv[2];
const outArg = process.argv[3];
if (!inputPath) {
  console.error("Uzycie: node tools/convert-replay.js <plik.hbr2|folder> [wyjscie.jsonl|--inspect]");
  console.error("  Folder: konwertuje wszystkie .hbr2 w srodku (pomija juz przekonwertowane).");
  process.exit(1);
}

// TRYB WSADOWY: folder z wieloma nagraniami - kazdy plik przerabiamy w osobnym
// procesie (czysty stan readera), sekwencyjnie, z pominieciem juz zrobionych.
if (fs.existsSync(inputPath) && fs.statSync(inputPath).isDirectory()) {
  const { spawnSync } = require("child_process");
  const files = fs.readdirSync(inputPath).filter((f) => /\.hbr2?$/i.test(f)).sort();
  if (files.length === 0) {
    console.error(`Brak plikow .hbr2 w ${inputPath}`);
    process.exit(1);
  }
  let done = 0, skipped = 0, failed = 0;
  for (const f of files) {
    const src = require("path").join(inputPath, f);
    const dst = src.replace(/\.hbr2?$/i, "") + ".jsonl";
    if (fs.existsSync(dst)) { skipped++; continue; }
    const res = spawnSync(process.execPath, [__filename, src, dst], { stdio: "inherit", timeout: 660000 });
    if (res.status === 0) done++; else { failed++; console.error(`BLAD przy ${f} (kod ${res.status})`); }
  }
  console.log(`\nGotowe: ${done} przekonwertowanych, ${skipped} pominietych (juz istnialy), ${failed} bledow.`);
  process.exit(failed > 0 ? 1 : 0);
}

const inspectMode = outArg === "--inspect";
const outPath = inspectMode ? null : (outArg || inputPath.replace(/\.hbr2?$/i, "") + ".jsonl");

const { Replay } = require("node-haxball")();
const data = fs.readFileSync(inputPath, null);

// aktualny stan klawiszy kazdego gracza (aktualizowany zdarzeniami)
const currentInputs = {};
const rows = [];
let inspected = false;

function firstDefined(...vals) {
  for (const v of vals) if (v !== undefined && v !== null) return v;
  return null;
}

function extractDisc(obj) {
  // rozne wersje node-haxball roznie nazywaja pola pozycji/predkosci
  const pos = firstDefined(obj?.pos, obj?.position);
  const speed = firstDefined(obj?.speed, obj?.velocity);
  if (!pos || !speed) return null;
  return { x: pos.x, y: pos.y, vx: speed.x, vy: speed.y };
}

const reader = Replay.read(data, {
  onPlayerInputChange: (id, value) => {
    currentInputs[id] = value;
  },
  onGameTick: () => {
    const state = reader.state;
    const gameState = reader.gameState;
    if (!state || !gameState) return;

    if (inspectMode && !inspected) {
      inspected = true;
      console.log("=== reader.state (klucze):", Object.keys(state));
      console.log("=== reader.gameState (klucze):", Object.keys(gameState));
      const players = firstDefined(state.players, state.playerList);
      if (players && players[0]) {
        console.log("=== przykladowy gracz (klucze):", Object.keys(players[0]));
        console.log("=== gracz.disc:", JSON.stringify(players[0].disc ?? null)?.slice(0, 300));
      }
      return;
    }

    const players = firstDefined(state.players, state.playerList) || [];
    const row = {
      f: reader.getCurrentFrameNo(),
      score: [
        firstDefined(gameState.redScore, state.redScore, 0),
        firstDefined(gameState.blueScore, state.blueScore, 0),
      ],
      ball: null,
      players: [],
    };

    for (const p of players) {
      const teamId = firstDefined(p.team?.id, p.team);
      if (teamId !== 1 && teamId !== 2) continue; // pomijamy widzow
      const disc = extractDisc(p.disc);
      if (!disc) continue;
      row.players.push({ id: p.id, team: teamId, ...disc, input: currentInputs[p.id] | 0 });
    }

    // pilka: pierwszy dysk stanu fizyki, ktory nie nalezy do zadnego gracza
    const discs = firstDefined(gameState.physicsState?.discs, gameState.discs);
    if (discs && discs.length > 0) {
      row.ball = extractDisc(discs[0]);
    }

    if (row.ball && row.players.length > 0) rows.push(row);
  },
  onEnd: () => finish(),
  onDestinationTimeReached: () => finish(),
}, {
  // wlasny scheduler - przelatujemy nagranie tak szybko, jak da rade CPU,
  // zamiast czekac w czasie rzeczywistym
  requestAnimationFrame: (cb) => setImmediate(cb),
  cancelAnimationFrame: (t) => clearImmediate(t),
});

let finished = false;
function finish() {
  if (finished) return;
  finished = true;
  if (!inspectMode) {
    fs.writeFileSync(outPath, rows.map((r) => JSON.stringify(r)).join("\n"));
    console.log(`Zapisano ${rows.length} tickow do ${outPath}`);
  } else {
    console.log("Inspekcja zakonczona.");
  }
  reader.destroy();
  process.exit(0);
}

console.log(`Nagranie: ${inputPath}, dlugosc: ${(reader.length() / 1000).toFixed(0)}s, klatek: ${reader.maxFrameNo}`);
reader.setSpeed(1000000);
reader.setCurrentFrameNo(reader.maxFrameNo);

// bezpiecznik - gdyby onEnd nie odpalil
setTimeout(() => { console.error("Timeout po 10 min - zapisuje co mam."); finish(); }, 600000);
