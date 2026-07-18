// UCZĄCY SIĘ 1v1: analogicznie do learner.js, ale każdy aktor hostuje mecz 1v1
// (2 boty), a doświadczenie OBU perspektyw (czerwony i niebieski) trafia do
// jednego, wspólnego modelu - stąd model uczy się z 2x większej ilości gry
// i jest od razu uniwersalny (gra dobrze z dowolnej strony boiska).
const { fork } = require("child_process");
const tf = require("@tensorflow/tfjs");
const fs = require("fs");
const { createModel } = require("./dqn-common-1v1.js");

// Tokeny trzymane osobno (tokens.js, niewersjonowany) - na każdej maszynie,
// która odpala trening, musi być własna kopia tego pliku ze świeżymi tokenami.
const { TOKENS_1V1: TOKENS } = require("./tokens.js");

// GAMMA=0.99 (bylo 0.95). Doswiadczenia licza sie teraz raz na decyzje (5 tickow,
// patrz actor-1v1.js), nie raz na tick - ale nawet tak, gol oddalony o ~2s (24
// decyzje) przy starym gamma=0.95 mial wartosc po zdyskontowaniu ~0.29^24≈0.0003.
// Przy 0.99 to 0.99^24≈0.79 - siec faktycznie "widzi" cel gry, nie tylko shaping.
const GAMMA = 0.99;
const BATCH_SIZE = 32;
// MAX_BUFFER=150000 (bylo 10000, czyli przy starym tempie ~8 SEKUND gry - dane
// ekstremalnie skorelowane, prosty przepis na zapadniecie sie polityki w jeden
// degenerat). Po agregacji do poziomu decyzji to ok. 10 minut zroznicowanej gry.
const MAX_BUFFER = 150000;
const WEIGHTS_PATH = "dqn-weights-1v1.json";
const PROGRESS_PATH = "training-progress-1v1.json";
const CHECKPOINTS_DIR = "checkpoints-1v1";
const LOG_PATH = "training-log-1v1.jsonl";
const SYNC_EVERY_MS = 2000;
// Bylo 4 - ale to bylo dostrojone do wolumenu doswiadczen PER TICK. Po agregacji
// do poziomu decyzji wolumen spada ~5x, wiec trenujemy czesciej per-doswiadczenie.
const TRAIN_EVERY_N_EXPERIENCES = 2;
const TOTAL_EPISODES = 30000; // łącznie, ze wszystkich aktorów, ZE WSZYSTKICH URUCHOMIEŃ - starczy na całą noc
// Wagi "najlepsze" zapisywaly sie tylko przy nowym rekordzie decisive-rate - zla,
// szumna metryka (liczy tez GOLE STRACONE jako "sukces", bo to tylko "czy padl
// gol", nie "czy wygral"). Ten checkpoint zapisuje wagi co jakis czas NIEZALEZNIE
// od rekordu, zeby crash nigdy nie kosztowal wiecej niz tyle epizodow.
const SAVE_CHECKPOINT_EVERY = 500;
// Co tyle epizodow zapisujemy DODATKOWO wersjonowana kopie wag (nie nadpisywana)
// do checkpoints-1v1/ - zeby dalo sie porownac/cofnac do modelu sprzed regresji.
const VERSION_CHECKPOINT_EVERY = 2000;
// STAŁE tempo zanikania epsilon na aktora - bylo 150 epizodow/aktora, czyli przy
// 10 aktorach ZALEDWIE ~1500 z 30000 epizodow (5% calego treningu!) mialo realna
// eksploracje. Przez pozostale 95% nocy boty grafy niemal deterministycznie tym,
// co zdazyly wypracowac w tym waskim oknie - bez szans na wyjscie ze zlego
// nawyku. 3000/aktora rozklada eksploracje na caly trening.
const EPSILON_DECAY_EPISODES_PER_ACTOR = 3000;
const EPSILON_MIN = 0.1;
// Jesli jedna akcja dominuje ponad tyle % wyborow w oknie - to sygnal, ze
// polityka sie zapada w jeden ruch (dokladnie to widzielismy wizualnie: boty
// stojace w miejscu/drgajace).
const ACTION_HISTOGRAM_WARN_THRESHOLD = 0.7;
const ACTION_NAMES = ["gora", "gora-prawo", "prawo", "dol-prawo", "dol", "dol-lewo", "lewo", "gora-lewo"];

if (!fs.existsSync(CHECKPOINTS_DIR)) fs.mkdirSync(CHECKPOINTS_DIR, { recursive: true });

let model = createModel();

function cloneModel(m) {
  const clone = createModel();
  clone.setWeights(m.getWeights().map((w) => w.clone()));
  return clone;
}

function loadWeightsIfExist() {
  if (!fs.existsSync(WEIGHTS_PATH)) {
    console.log("[uczący się 1v1] brak zapisanych wag - start od losowej sieci");
    return;
  }
  try {
    const saved = JSON.parse(fs.readFileSync(WEIGHTS_PATH, "utf8"));
    model.setWeights(saved.map((w) => tf.tensor(w)));
    console.log("[uczący się 1v1] wczytano zapisane wagi - kontynuuję zamiast zaczynać od zera");
  } catch (e) {
    console.log("[uczący się 1v1] zapisane wagi nie pasują do architektury - start od losowej sieci");
  }
}

loadWeightsIfExist();
let targetModel = cloneModel(model);

let episodesBefore = 0;
let bestAvgReward = -Infinity;
if (fs.existsSync(PROGRESS_PATH)) {
  try {
    const progress = JSON.parse(fs.readFileSync(PROGRESS_PATH, "utf8"));
    episodesBefore = progress.totalEpisodes || 0;
    bestAvgReward = typeof progress.bestAvgReward === "number" ? progress.bestAvgReward : -Infinity;
    console.log(`[uczący się 1v1] wczytano postęp: ${episodesBefore} epizodów już za nami, najlepsza śr. nagroda dotąd ${bestAvgReward === -Infinity ? "brak" : bestAvgReward.toFixed(1)}`);
  } catch (e) {
    console.log("[uczący się 1v1] nie udało się wczytać pliku postępu - liczę od zera");
  }
}

function saveProgress(totalEpisodesNow) {
  fs.writeFileSync(PROGRESS_PATH, JSON.stringify({ totalEpisodes: totalEpisodesNow, bestAvgReward }));
}

function saveWeights() {
  fs.writeFileSync(WEIGHTS_PATH, JSON.stringify(model.getWeights().map((w) => w.arraySync())));
}

function saveVersionedCheckpoint(episode) {
  const path = `${CHECKPOINTS_DIR}/dqn-weights-1v1-ep${String(episode).padStart(6, "0")}.json`;
  fs.writeFileSync(path, JSON.stringify(model.getWeights().map((w) => w.arraySync())));
  console.log(`[uczący się 1v1]  -> zapisano wersjonowany checkpoint: ${path}`);
}

function logEpisode(record) {
  fs.appendFileSync(LOG_PATH, JSON.stringify(record) + "\n");
}

const replayBuffer = [];
let experienceCountSinceTrain = 0;
const episodeResults = []; // { winner: 1 | 2 | null }
let training = false;
let lastCheckpointEpisode = episodesBefore;
let lastVersionedEpisode = episodesBefore;

// Telemetria treningu - zeby wiedziec, ile faktycznie krokow gradientu sie
// odbywa (trainStep potrafi po cichu POMIJAC trening, jesli poprzedni fit()
// jeszcze trwa), a nie zgadywac na podstawie TRAIN_EVERY_N_EXPERIENCES.
let trainStepsAttempted = 0;
let trainStepsExecuted = 0;
let trainStepsSkipped = 0;
let lossSum = 0;
let lossCount = 0;

async function trainStep() {
  trainStepsAttempted++;
  if (training) { trainStepsSkipped++; return; }
  if (replayBuffer.length < BATCH_SIZE) { trainStepsSkipped++; return; }
  training = true;
  try {
    const batch = [];
    for (let i = 0; i < BATCH_SIZE; i++) {
      batch.push(replayBuffer[Math.floor(Math.random() * replayBuffer.length)]);
    }
    const states = batch.map((b) => b.features);
    const nextStates = batch.map((b) => b.nextFeatures);
    const qCurrent = model.predict(tf.tensor2d(states)).arraySync();
    const qNextOnline = model.predict(tf.tensor2d(nextStates)).arraySync();
    const qNextTarget = targetModel.predict(tf.tensor2d(nextStates)).arraySync();
    const targets = qCurrent.map((qValues, i) => {
      const { action, reward, done } = batch[i];
      const target = qValues.slice();
      if (done) {
        target[action] = reward;
      } else {
        let bestNextAction = 0;
        for (let a = 1; a < qNextOnline[i].length; a++) {
          if (qNextOnline[i][a] > qNextOnline[i][bestNextAction]) bestNextAction = a;
        }
        target[action] = reward + GAMMA * qNextTarget[i][bestNextAction];
      }
      return target;
    });
    const history = await model.fit(tf.tensor2d(states), tf.tensor2d(targets), { epochs: 1, verbose: 0 });
    trainStepsExecuted++;
    const loss = history.history.loss ? history.history.loss[0] : null;
    if (typeof loss === "number" && Number.isFinite(loss)) {
      lossSum += loss;
      lossCount++;
    }
  } finally {
    training = false;
  }
}

let stopped = false;

// Bufory telemetrii dla okna 20 epizodow (do logu zbiorczego w konsoli).
let windowRedTouches = 0, windowBlueTouches = 0;
let windowRedAvgSpeedSum = 0, windowBlueAvgSpeedSum = 0;
let windowRedRewardSum = 0, windowBlueRewardSum = 0;
const windowRedActionCounts = new Array(8).fill(0);
const windowBlueActionCounts = new Array(8).fill(0);

const actors = TOKENS.map((token, i) => {
  const startEpisode = Math.floor(episodesBefore / TOKENS.length);
  const child = fork("actor-1v1.js", [
    token,
    String(i),
    String(EPSILON_DECAY_EPISODES_PER_ACTOR),
    String(startEpisode),
    String(EPSILON_MIN),
  ], { silent: true });

  child.stdout.on("data", (d) => {
    d.toString().split("\n").filter(Boolean).forEach((l) => console.log(`[mecz ${i}] ${l}`));
  });
  child.stderr.on("data", (d) => {
    d.toString().split("\n").filter(Boolean).forEach((l) => console.error(`[mecz ${i}] BŁĄD: ${l}`));
  });

  // Bez tego, jeśli aktor padnie (crash / zamknięty kanał IPC), kolejna próba
  // child.send() w setInterval niżej rzuca nieobsłużony błąd i ubija CAŁY
  // proces learnera - razem z pozostałymi, wciąż żywymi aktorami. Teraz
  // martwy aktor tylko przestaje uczestniczyć, reszta gra dalej.
  child.on("error", (err) => {
    console.error(`[mecz ${i}] proces zerwał połączenie: ${err.message}`);
  });
  child.on("exit", (code, signal) => {
    console.error(`[mecz ${i}] proces zakończony (kod ${code}${signal ? `, sygnał ${signal}` : ""})`);
  });

  child.on("message", async (msg) => {
    if (stopped) return;

    if (msg.type === "experience") {
      replayBuffer.push(msg.data);
      if (replayBuffer.length > MAX_BUFFER) replayBuffer.shift();
      experienceCountSinceTrain++;
      if (experienceCountSinceTrain >= TRAIN_EVERY_N_EXPERIENCES) {
        experienceCountSinceTrain = 0;
        await trainStep();
      }
    } else if (msg.type === "episode_done") {
      episodeResults.push(msg.winner);
      const totalEpisodesNow = episodesBefore + episodeResults.length;
      const t = msg.telemetry || {};

      logEpisode({
        episode: totalEpisodesNow,
        actor: i,
        winner: msg.winner,
        ...t,
      });

      windowRedTouches += t.redTouches || 0;
      windowBlueTouches += t.blueTouches || 0;
      windowRedAvgSpeedSum += t.redAvgSpeed || 0;
      windowBlueAvgSpeedSum += t.blueAvgSpeed || 0;
      windowRedRewardSum += t.redRewardSum || 0;
      windowBlueRewardSum += t.blueRewardSum || 0;
      (t.redActionCounts || []).forEach((c, idx) => { windowRedActionCounts[idx] += c; });
      (t.blueActionCounts || []).forEach((c, idx) => { windowBlueActionCounts[idx] += c; });

      if (episodeResults.length % 20 === 0) {
        const last20 = episodeResults.slice(-20);
        const redWins = last20.filter((w) => w === 1).length;
        const blueWins = last20.filter((w) => w === 2).length;
        const draws = last20.filter((w) => w === null).length;
        const decisiveRate = (redWins + blueWins) / 20;
        const avgReward = (windowRedRewardSum + windowBlueRewardSum) / (2 * 20);
        const avgSpeed = (windowRedAvgSpeedSum + windowBlueAvgSpeedSum) / (2 * 20);

        console.log(`[uczący się 1v1] epizody łącznie: ${totalEpisodesNow}, czerwoni ${redWins}/20, niebiescy ${blueWins}/20, remis ${draws}/20`);
        console.log(`[uczący się 1v1]   śr. nagroda/epizod: ${avgReward.toFixed(1)}, śr. prędkość: ${avgSpeed.toFixed(2)}, dotknięcia: czerw ${windowRedTouches} nieb ${windowBlueTouches}`);

        const executedThisWindow = trainStepsExecuted;
        const skippedThisWindow = trainStepsSkipped;
        const avgLoss = lossCount > 0 ? lossSum / lossCount : null;
        console.log(`[uczący się 1v1]   trening: bufor ${replayBuffer.length}/${MAX_BUFFER}, kroki gradientu wykonane ${executedThisWindow} / pominięte ${skippedThisWindow} (łącznie od startu procesu), śr. loss ${avgLoss === null ? "brak" : avgLoss.toFixed(4)}`);

        const redTotal = windowRedActionCounts.reduce((a, b) => a + b, 0) || 1;
        const blueTotal = windowBlueActionCounts.reduce((a, b) => a + b, 0) || 1;
        const redMaxIdx = windowRedActionCounts.indexOf(Math.max(...windowRedActionCounts));
        const blueMaxIdx = windowBlueActionCounts.indexOf(Math.max(...windowBlueActionCounts));
        const redMaxShare = windowRedActionCounts[redMaxIdx] / redTotal;
        const blueMaxShare = windowBlueActionCounts[blueMaxIdx] / blueTotal;
        if (redMaxShare > ACTION_HISTOGRAM_WARN_THRESHOLD || blueMaxShare > ACTION_HISTOGRAM_WARN_THRESHOLD) {
          console.log(`[uczący się 1v1]   UWAGA: polityka może się zapadać - czerwoni akcja "${ACTION_NAMES[redMaxIdx]}" ${(redMaxShare * 100).toFixed(0)}%, niebiescy akcja "${ACTION_NAMES[blueMaxIdx]}" ${(blueMaxShare * 100).toFixed(0)}%`);
        }

        targetModel = cloneModel(model);

        if (avgReward > bestAvgReward) {
          bestAvgReward = avgReward;
          saveWeights();
          lastCheckpointEpisode = totalEpisodesNow;
          console.log(`[uczący się 1v1]  -> nowy rekord śr. nagrody (${avgReward.toFixed(1)}), zapisuję wagi`);
        } else if (totalEpisodesNow - lastCheckpointEpisode >= SAVE_CHECKPOINT_EVERY) {
          saveWeights();
          lastCheckpointEpisode = totalEpisodesNow;
          console.log(`[uczący się 1v1]  -> checkpoint co ${SAVE_CHECKPOINT_EVERY} epizodów (bez nowego rekordu), zapisuję wagi`);
        }
        if (totalEpisodesNow - lastVersionedEpisode >= VERSION_CHECKPOINT_EVERY) {
          saveVersionedCheckpoint(totalEpisodesNow);
          lastVersionedEpisode = totalEpisodesNow;
        }
        saveProgress(totalEpisodesNow);

        // reset okna
        windowRedTouches = 0; windowBlueTouches = 0;
        windowRedAvgSpeedSum = 0; windowBlueAvgSpeedSum = 0;
        windowRedRewardSum = 0; windowBlueRewardSum = 0;
        windowRedActionCounts.fill(0);
        windowBlueActionCounts.fill(0);
        lossSum = 0; lossCount = 0;
      }

      if (totalEpisodesNow >= TOTAL_EPISODES && !stopped) {
        stopped = true;
        console.log(`[uczący się 1v1] osiągnięto ${TOTAL_EPISODES} epizodów łącznie - kończę.`);
        saveWeights();
        saveVersionedCheckpoint(totalEpisodesNow);
        saveProgress(totalEpisodesNow);
        actors.forEach(({ child }) => { if (child.connected) child.send({ type: "stop" }); });
        setTimeout(() => process.exit(0), 500);
      }
    }
  });

  return { child, index: i };
});

setInterval(() => {
  if (stopped) return;
  const weights = model.getWeights().map((w) => w.arraySync());
  actors.forEach(({ child }) => { if (child.connected) child.send({ type: "weights", weights }); });
}, SYNC_EVERY_MS);

console.log(`[uczący się 1v1] wystartowano z ${TOKENS.length} aktorami (${TOKENS.length} równoległych meczów 1v1)`);
