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

const GAMMA = 0.95;
const BATCH_SIZE = 32;
const MAX_BUFFER = 10000;
const WEIGHTS_PATH = "dqn-weights-1v1.json";
const PROGRESS_PATH = "training-progress-1v1.json";
const SYNC_EVERY_MS = 2000;
const TRAIN_EVERY_N_EXPERIENCES = 4;
const TOTAL_EPISODES = 30000; // łącznie, ze wszystkich aktorów, ZE WSZYSTKICH URUCHOMIEŃ - starczy na całą noc
// Wagi zapisują się normalnie tylko przy nowym rekordzie decisive-rate - jeśli
// wynik utknie w miejscu (tak jak się zdarzyło - 50% od epizodu ~4840 do 30000),
// zapis w ogóle się nie odpala przez resztę nocy. Gdyby proces padł przed
// końcem, stracilibyśmy cały ten postęp. Ten checkpoint zapisuje wagi co jakiś
// czas NIEZALEŻNIE od rekordu, żeby crash nigdy nie kosztował więcej niż tyle
// epizodów, ile wynosi ten interwał.
const SAVE_CHECKPOINT_EVERY = 500;
// STAŁE tempo zanikania epsilon na aktora - nie zależy od TOTAL_EPISODES (patrz
// haxball-ai-weight-migration w pamięci projektu - to samo co poprawka w learner.js)
const EPSILON_DECAY_EPISODES_PER_ACTOR = 150;

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
let bestSuccessRate = -1;
if (fs.existsSync(PROGRESS_PATH)) {
  try {
    const progress = JSON.parse(fs.readFileSync(PROGRESS_PATH, "utf8"));
    episodesBefore = progress.totalEpisodes || 0;
    bestSuccessRate = typeof progress.bestSuccessRate === "number" ? progress.bestSuccessRate : -1;
    console.log(`[uczący się 1v1] wczytano postęp: ${episodesBefore} epizodów już za nami, najlepszy wynik dotąd ${(bestSuccessRate * 100).toFixed(0)}%`);
  } catch (e) {
    console.log("[uczący się 1v1] nie udało się wczytać pliku postępu - liczę od zera");
  }
}

function saveProgress(totalEpisodesNow) {
  fs.writeFileSync(PROGRESS_PATH, JSON.stringify({ totalEpisodes: totalEpisodesNow, bestSuccessRate }));
}

function saveWeights() {
  fs.writeFileSync(WEIGHTS_PATH, JSON.stringify(model.getWeights().map((w) => w.arraySync())));
}

const replayBuffer = [];
let experienceCountSinceTrain = 0;
const episodeResults = []; // { winner: 1 | 2 | null }
let training = false;
let lastCheckpointEpisode = episodesBefore;

async function trainStep() {
  if (training) return;
  if (replayBuffer.length < BATCH_SIZE) return;
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
    await model.fit(tf.tensor2d(states), tf.tensor2d(targets), { epochs: 1, verbose: 0 });
  } finally {
    training = false;
  }
}

let stopped = false;

const actors = TOKENS.map((token, i) => {
  const startEpisode = Math.floor(episodesBefore / TOKENS.length);
  const child = fork("actor-1v1.js", [token, String(i), String(EPSILON_DECAY_EPISODES_PER_ACTOR), String(startEpisode)], { silent: true });

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

      if (episodeResults.length % 20 === 0) {
        const last20 = episodeResults.slice(-20);
        const redWins = last20.filter((w) => w === 1).length;
        const blueWins = last20.filter((w) => w === 2).length;
        const draws = last20.filter((w) => w === null).length;
        const decisiveRate = (redWins + blueWins) / 20;
        console.log(`[uczący się 1v1] epizody łącznie: ${totalEpisodesNow}, czerwoni ${redWins}/20, niebiescy ${blueWins}/20, remis ${draws}/20`);

        targetModel = cloneModel(model);

        if (decisiveRate > bestSuccessRate) {
          bestSuccessRate = decisiveRate;
          saveWeights();
          lastCheckpointEpisode = totalEpisodesNow;
          console.log(`[uczący się 1v1]  -> nowy rekord (${(decisiveRate * 100).toFixed(0)}% meczów z golem), zapisuję wagi`);
        } else if (totalEpisodesNow - lastCheckpointEpisode >= SAVE_CHECKPOINT_EVERY) {
          saveWeights();
          lastCheckpointEpisode = totalEpisodesNow;
          console.log(`[uczący się 1v1]  -> checkpoint co ${SAVE_CHECKPOINT_EVERY} epizodów (bez nowego rekordu), zapisuję wagi`);
        }
        saveProgress(totalEpisodesNow);
      }

      if (totalEpisodesNow >= TOTAL_EPISODES && !stopped) {
        stopped = true;
        console.log(`[uczący się 1v1] osiągnięto ${TOTAL_EPISODES} epizodów łącznie - kończę.`);
        saveWeights();
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
