// UCZĄCY SIĘ: nie łączy się z żadnym pokojem HaxBall, tylko odbiera doświadczenie
// od aktorów (actor.js) przez IPC, trenuje jeden wspólny model, i okresowo rozsyła
// świeże wagi z powrotem do wszystkich aktorów. Cała ciężka matematyka (model.fit)
// dzieje się tylko tutaj, w jednym procesie, więc nie konkuruje z fizyką pokoi o wątek JS.
const { fork } = require("child_process");
const tf = require("@tensorflow/tfjs");
const fs = require("fs");
const { createModel } = require("./dqn-common.js");

// Tokeny trzymane osobno (tokens.js, niewersjonowany).
const { TOKENS_1V1: TOKENS } = require("./tokens.js");

const GAMMA = 0.95;
const BATCH_SIZE = 32;
const MAX_BUFFER = 5000;
const WEIGHTS_PATH = "dqn-weights-goal.json";
const PROGRESS_PATH = "training-progress.json"; // pamięta epsilon/postęp MIĘDZY uruchomieniami learner.js
const SYNC_EVERY_MS = 2000; // co ile rozsyłamy świeże wagi do aktorów
const TRAIN_EVERY_N_EXPERIENCES = 4; // co ile nowych doświadczeń robimy jeden krok treningu
const TOTAL_EPISODES = 15000; // łącznie, licząc epizody ze wszystkich aktorów razem, ZE WSZYSTKICH URUCHOMIEŃ

let model = createModel();
loadWeightsIfExist();
let targetModel = cloneModel(model);

// Wczytujemy, ile epizodów już łącznie rozegraliśmy i jaki był najlepszy wynik -
// bez tego każdy restart resetowałby epsilon aktorów z powrotem do maksimum.
let episodesBefore = 0;
let bestSuccessRate = -1;
if (fs.existsSync(PROGRESS_PATH)) {
  try {
    const progress = JSON.parse(fs.readFileSync(PROGRESS_PATH, "utf8"));
    episodesBefore = progress.totalEpisodes || 0;
    bestSuccessRate = typeof progress.bestSuccessRate === "number" ? progress.bestSuccessRate : -1;
    console.log(`[uczący się] wczytano postęp: ${episodesBefore} epizodów już za nami, najlepszy wynik dotąd ${(bestSuccessRate * 100).toFixed(0)}%`);
  } catch (e) {
    console.log("[uczący się] nie udało się wczytać pliku postępu - liczę od zera");
  }
}

function saveProgress(totalEpisodesNow) {
  fs.writeFileSync(PROGRESS_PATH, JSON.stringify({ totalEpisodes: totalEpisodesNow, bestSuccessRate }));
}

const replayBuffer = [];
let experienceCountSinceTrain = 0;
const episodeResults = [];
let training = false;

function cloneModel(m) {
  const clone = createModel();
  clone.setWeights(m.getWeights().map((w) => w.clone()));
  return clone;
}

function loadWeightsIfExist() {
  if (!fs.existsSync(WEIGHTS_PATH)) {
    console.log("[uczący się] brak zapisanych wag - start od losowej sieci");
    return;
  }
  try {
    const saved = JSON.parse(fs.readFileSync(WEIGHTS_PATH, "utf8"));
    model.setWeights(saved.map((w) => tf.tensor(w)));
    console.log("[uczący się] wczytano zapisane wagi - kontynuuję zamiast zaczynać od zera");
  } catch (e) {
    console.log("[uczący się] zapisane wagi nie pasują do architektury - start od losowej sieci");
  }
}

function saveWeights() {
  const weights = model.getWeights().map((w) => w.arraySync());
  fs.writeFileSync(WEIGHTS_PATH, JSON.stringify(weights));
}

async function trainStep() {
  if (training) return; // nie nakładamy kolejnego treningu, jeśli poprzedni jeszcze trwa
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
    // Double DQN: sieć "online" (model) wybiera najlepszą akcję dla następnego stanu,
    // a sieć "target" tylko ją wycenia. Rozdzielenie wyboru od wyceny zmniejsza
    // tendencję zwykłego DQN do przeszacowywania wartości akcji.
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

// STAŁE tempo zanikania epsilon na aktora - NIE zależy od TOTAL_EPISODES/liczby aktorów.
// Gdyby zależało (jak poprzednio), podbicie budżetu epizodów samo w sobie "cofałoby"
// epsilon z powrotem w górę dla już wytrenowanych aktorów - dokładnie to się właśnie stało.
const EPSILON_DECAY_EPISODES_PER_ACTOR = 150;

const actors = TOKENS.map((token, i) => {
  // ile epizodów TEN aktor już rozegrał w poprzednich uruchomieniach (w przybliżeniu,
  // zakładając równy podział) - żeby jego epsilon kontynuował zanikanie, a nie startował od 0.5
  const startEpisode = Math.floor(episodesBefore / TOKENS.length);
  const child = fork("actor.js", [token, String(i), String(EPSILON_DECAY_EPISODES_PER_ACTOR), String(startEpisode)], { silent: true });

  child.stdout.on("data", (d) => {
    d.toString().split("\n").filter(Boolean).forEach((l) => console.log(`[bot ${i}] ${l}`));
  });
  child.stderr.on("data", (d) => {
    d.toString().split("\n").filter(Boolean).forEach((l) => console.error(`[bot ${i}] BŁĄD: ${l}`));
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
      episodeResults.push(msg.success);
      const totalEpisodesNow = episodesBefore + episodeResults.length;

      if (episodeResults.length % 20 === 0) {
        const last20 = episodeResults.slice(-20);
        const successRate = last20.filter(Boolean).length / last20.length;
        console.log(`[uczący się] epizody łącznie: ${totalEpisodesNow}, skuteczność (ostatnie 20): ${(successRate * 100).toFixed(0)}%`);

        targetModel = cloneModel(model);

        if (successRate > bestSuccessRate) {
          bestSuccessRate = successRate;
          saveWeights();
          console.log(`[uczący się]  -> nowy rekord (${(bestSuccessRate * 100).toFixed(0)}%), zapisuję wagi`);
        }
        saveProgress(totalEpisodesNow);
      }

      if (totalEpisodesNow >= TOTAL_EPISODES && !stopped) {
        stopped = true;
        console.log(`[uczący się] osiągnięto ${TOTAL_EPISODES} epizodów łącznie (ze wszystkich botów, ze wszystkich uruchomień) - kończę.`);
        saveWeights();
        saveProgress(totalEpisodesNow);
        actors.forEach(({ child }) => child.send({ type: "stop" }));
        setTimeout(() => process.exit(0), 500);
      }
    }
  });

  return { child, index: i };
});

// okresowo rozsyłamy świeże wagi do wszystkich aktorów, żeby ich lokalne kopie
// nie odjeżdżały zbyt daleko od tego, czego się właśnie nauczyliśmy
setInterval(() => {
  if (stopped) return;
  const weights = model.getWeights().map((w) => w.arraySync());
  actors.forEach(({ child }) => child.send({ type: "weights", weights }));
}, SYNC_EVERY_MS);

console.log(`[uczący się] wystartowano z ${TOKENS.length} aktorami`);
