// Sortowanie nagran .hbr2 wg mapy: czyta nazwe stadionu z kazdego pliku
// i przenosi go do podfolderu o nazwie mapy. Idealne do hurtowo pobranych
// nagran (np. z MrREPLAY), gdzie mapy sa wymieszane.
//
// Uzycie:
//   node tools/sort-replays.js replays/            # sortuje
//   node tools/sort-replays.js replays/ --dry-run  # tylko pokazuje co by zrobil
//
// Przenosi tez odpowiadajace pliki .jsonl (jesli istnieja).
const fs = require("fs");
const path = require("path");

const dir = process.argv[2];
const dryRun = process.argv.includes("--dry-run");
if (!dir || !fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) {
  console.error("Uzycie: node tools/sort-replays.js <folder> [--dry-run]");
  process.exit(1);
}

const { Replay } = require("node-haxball")();

function sanitize(name) {
  return (name || "nieznana-mapa")
    .replace(/[^a-zA-Z0-9 _.-]/g, "")
    .trim()
    .replace(/\s+/g, "-")
    .slice(0, 60) || "nieznana-mapa";
}

const files = fs.readdirSync(dir).filter((f) => /\.hbr2?$/i.test(f));
if (files.length === 0) {
  console.error(`Brak plikow .hbr2 w ${dir}`);
  process.exit(1);
}

const counts = {};
let failed = 0;

for (const f of files) {
  const src = path.join(dir, f);
  let stadium = null;
  try {
    const data = fs.readFileSync(src, null);
    const rd = Replay.readAll(data);
    stadium = rd.roomData?.stadium?.name ?? null;
  } catch (e) {
    failed++;
    console.error(`  BLAD odczytu ${f} - pomijam (plik uszkodzony/urwany?)`);
    continue;
  }
  const folder = sanitize(stadium);
  counts[folder] = (counts[folder] || 0) + 1;
  const destDir = path.join(dir, folder);
  const dst = path.join(destDir, f);
  console.log(`${f}  ->  ${folder}/  (stadion: ${stadium ?? "?"})`);
  if (!dryRun) {
    fs.mkdirSync(destDir, { recursive: true });
    fs.renameSync(src, dst);
    const jsonl = src.replace(/\.hbr2?$/i, "") + ".jsonl";
    if (fs.existsSync(jsonl)) {
      fs.renameSync(jsonl, path.join(destDir, path.basename(jsonl)));
    }
  }
}

console.log("");
console.log("Podsumowanie wg map:");
for (const [folder, n] of Object.entries(counts).sort((a, b) => b[1] - a[1])) {
  console.log(`  ${folder}: ${n}`);
}
if (failed > 0) console.log(`  (uszkodzonych/nieczytelnych: ${failed})`);
if (dryRun) console.log("\n(dry-run - nic nie zostalo przeniesione)");
