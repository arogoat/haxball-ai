// Konwerter nagran haxballa (.hbr2) -> dane treningowe (JSONL, 1 linia = 1 tick).
// Fundament pod imitation learning (nauka z gier dobrych graczy).
//
// Uzycie (na serwerze, w katalogu glownym repo - potrzebuje node_modules):
//   node tools/convert-replay.js sciezka/do/nagrania.hbr2 [wyjscie.jsonl]
//   node tools/convert-replay.js replays/            # wszystkie .hbr2 w folderze
//   node tools/convert-replay.js plik.hbr2 --inspect # podglad struktury pliku
//
// Format wyjscia:
//   linia 1: { meta: { stadium, totalFrames, players: [{id, name, team}] } }
//   kolejne: { f, score: [r, b], ball: {x,y,vx,vy},
//              players: [{ id, team, x, y, vx, vy, input }] }
//   input = bitmaska klawiszy (1=gora, 2=dol, 4=lewo, 8=prawo, 16=kop).
//
// Przetestowane end-to-end na prawdziwych nagraniach (w tym Futsal x7, 16 graczy).
const fs = require("fs");

const inputPath = process.argv[2];
const outArg = process.argv[3];
if (!inputPath) {
  console.error("Uzycie: node tools/convert-replay.js <plik.hbr2|folder> [wyjscie.jsonl|--inspect]");
  process.exit(1);
}

// TRYB WSADOWY: folder z wieloma nagraniami - kazdy plik w osobnym procesie,
// sekwencyjnie, z pominieciem juz przekonwertowanych.
if (fs.existsSync(inputPath) && fs.statSync(inputPath).isDirectory()) {
  const { spawnSync } = require("child_process");
  const path = require("path");
  const files = fs.readdirSync(inputPath).filter((f) => /\.hbr2?$/i.test(f)).sort();
  if (files.length === 0) {
    console.error(`Brak plikow .hbr2 w ${inputPath}`);
    process.exit(1);
  }
  let done = 0, skipped = 0, failed = 0;
  for (const f of files) {
    const src = path.join(inputPath, f);
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

// TRYB INSPEKCJI: synchroniczny parse przez Replay.readAll - struktura na stol.
if (inspectMode) {
  const rd = Replay.readAll(data);
  console.log("totalFrames:", rd.totalFrames, "version:", rd.version);
  console.log("stadion:", rd.roomData?.stadium?.name ?? "(?)");
  console.log("graczy:", rd.roomData?.players?.length ?? 0);
  console.log("zdarzen:", rd.events?.length ?? 0);
  process.exit(0);
}

const rows = [];
let meta = null;
let lastProgress = 0;

const reader = Replay.read(data, {
  onGameTick: () => {
    const gameState = reader.gameState;
    const state = reader.state;
    if (!gameState || !state) return;

    if (!meta) {
      meta = {
        stadium: state.stadium?.name ?? gameState.stadium?.name ?? null,
        totalFrames: reader.maxFrameNo,
        players: (state.players || [])
          .filter((p) => p.team?.id === 1 || p.team?.id === 2)
          .map((p) => ({ id: p.id, name: p.name, team: p.team.id })),
      };
    }

    const f = reader.getCurrentFrameNo();
    if (f - lastProgress >= 20000) {
      lastProgress = f;
      console.log(`  ...klatka ${f}/${reader.maxFrameNo}`);
    }

    const ball = gameState.physicsState?.discs?.[0];
    if (!ball?.pos) return;

    const row = {
      f,
      score: [gameState.redScore | 0, gameState.blueScore | 0],
      ball: { x: ball.pos.x, y: ball.pos.y, vx: ball.speed.x, vy: ball.speed.y },
      players: [],
    };
    for (const p of state.players || []) {
      const teamId = p.team?.id;
      if (teamId !== 1 && teamId !== 2) continue; // widzowie
      if (!p.disc?.pos) continue; // gracz bez dysku (np. wlasnie dolaczyl)
      row.players.push({
        id: p.id, team: teamId,
        x: p.disc.pos.x, y: p.disc.pos.y,
        vx: p.disc.speed.x, vy: p.disc.speed.y,
        input: p.input | 0,
      });
    }
    if (row.players.length > 0) rows.push(row);
  },
}, {
  // wlasny scheduler - nagranie przelatuje tak szybko jak CPU pozwala
  requestAnimationFrame: (cb) => setImmediate(cb),
  cancelAnimationFrame: (t) => clearImmediate(t),
});

let finished = false;
function finish() {
  if (finished) return;
  finished = true;
  const out = [JSON.stringify({ meta: meta ?? { stadium: null, totalFrames: reader.maxFrameNo, players: [] } })];
  for (const r of rows) out.push(JSON.stringify(r));
  fs.writeFileSync(outPath, out.join("\n"));
  console.log(`Zapisano ${rows.length} tickow do ${outPath} (stadion: ${meta?.stadium ?? "?"})`);
  try { reader.destroy(); } catch (e) { /* ignore */ }
  process.exit(0);
}

reader.onEnd = () => finish();

console.log(`Nagranie: ${inputPath}, dlugosc: ${(reader.length() / 1000).toFixed(0)}s, klatek: ${reader.maxFrameNo}`);
reader.setSpeed(1000000);

// bezpiecznik - gdyby onEnd nie odpalil
setTimeout(() => { console.error("Timeout po 10 min - zapisuje co mam."); finish(); }, 600000);
