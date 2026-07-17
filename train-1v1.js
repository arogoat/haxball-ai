// Trening 1v1: jeden pokój, dwa boty (czerwony, niebieski), ale JEDEN wspólny,
// uniwersalny model - każdy bot używa tej samej sieci, tylko z danymi liczonymi
// względem siebie (getFeatures(state, team) - patrz dqn-common-1v1.js). Doświadczenie
// obu botów trafia do jednego, wspólnego bufora - model uczy się z dwa razy większej
// ilości gry w tym samym czasie, i od razu jest gotowy do gry z dowolnej strony boiska.
const { createEnv1v1 } = require("./env-1v1.js");
const tf = require("@tensorflow/tfjs");
const fs = require("fs");
const { ACTIONS, getFeatures, computeReward, createModel } = require("./dqn-common-1v1.js");

const WEIGHTS_PATH = "dqn-weights-1v1.json";

function cloneModel(m) {
  const clone = createModel();
  clone.setWeights(m.getWeights().map((w) => w.clone()));
  return clone;
}

function loadWeightsIfExist(model, path) {
  if (!fs.existsSync(path)) {
    console.log(`Brak ${path} - start od losowej sieci.`);
    return;
  }
  try {
    const saved = JSON.parse(fs.readFileSync(path, "utf8"));
    model.setWeights(saved.map((w) => tf.tensor(w)));
    console.log(`Wczytano ${path} - kontynuuję.`);
  } catch (e) {
    console.log(`${path} nie pasuje do architektury - start od losowej sieci.`);
  }
}

function saveWeights(model, path) {
  fs.writeFileSync(path, JSON.stringify(model.getWeights().map((w) => w.arraySync())));
}

function chooseAction(model, features, epsilon) {
  if (Math.random() < epsilon) return Math.floor(Math.random() * ACTIONS.length);
  const qValues = tf.tidy(() => model.predict(tf.tensor2d([features])).dataSync());
  let best = 0;
  for (let a = 1; a < qValues.length; a++) if (qValues[a] > qValues[best]) best = a;
  return best;
}

const model = createModel();
loadWeightsIfExist(model, WEIGHTS_PATH);
let targetModel = cloneModel(model);

const replayBuffer = [];
const MAX_BUFFER = 10000; // większy niż przy 1 bocie, bo teraz wpada 2x tyle doświadczenia
const BATCH_SIZE = 32;
const GAMMA = 0.95;

function remember(features, action, reward, nextFeatures, done) {
  replayBuffer.push({ features, action, reward, nextFeatures, done });
  if (replayBuffer.length > MAX_BUFFER) replayBuffer.shift();
}

async function trainOnBatch() {
  if (replayBuffer.length < BATCH_SIZE) return;
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
}

const TOTAL_EPISODES = 3000;
const MAX_STEPS = 900;
const ACTION_REPEAT = 5;
const EPSILON_DECAY_EPISODES = 500;

async function train(env) {
  const history = [];
  let bestSuccessRate = -1; // liczona jako % epizodów, które NIE skończyły się remisem (ktoś strzelił gola)

  for (let ep = 0; ep < TOTAL_EPISODES; ep++) {
    const progress = Math.min(1, ep / EPSILON_DECAY_EPISODES);
    const epsilon = Math.max(0.05, 0.5 - 0.45 * progress);

    let state = env.reset();
    let redFeatures = getFeatures(state, 1);
    let blueFeatures = getFeatures(state, 2);
    let redAction = chooseAction(model, redFeatures, epsilon);
    let blueAction = chooseAction(model, blueFeatures, epsilon);
    let winner = null;

    for (let step = 0; step < MAX_STEPS; step++) {
      if (step % ACTION_REPEAT === 0) {
        redAction = chooseAction(model, redFeatures, epsilon);
        blueAction = chooseAction(model, blueFeatures, epsilon);
      }
      const [redDirX, redDirY] = ACTIONS[redAction];
      const [blueDirX, blueDirY] = ACTIONS[blueAction];

      const redDist = Math.hypot(state.ballX - state.redX, state.ballY - state.redY);
      const blueDist = Math.hypot(state.ballX - state.blueX, state.ballY - state.blueY);
      const redKick = redDist < (state.ballRadius + state.redRadius + 5);
      const blueKick = blueDist < (state.ballRadius + state.blueRadius + 5);

      const newState = await env.step(redDirX, redDirY, redKick, blueDirX, blueDirY, blueKick);

      const newRedFeatures = getFeatures(newState, 1);
      const newBlueFeatures = getFeatures(newState, 2);
      const redResult = computeReward(state, newState, 1);
      const blueResult = computeReward(state, newState, 2);

      // obie perspektywy (czerwony i niebieski) trafiają do TEGO SAMEGO bufora -
      // to jest właśnie źródło "uniwersalności" tego modelu
      remember(redFeatures, redAction, redResult.reward, newRedFeatures, redResult.done);
      remember(blueFeatures, blueAction, blueResult.reward, newBlueFeatures, blueResult.done);

      await trainOnBatch();

      state = newState;
      redFeatures = newRedFeatures;
      blueFeatures = newBlueFeatures;

      if (newState.scoredBy) { winner = newState.scoredBy; break; }
    }

    history.push({ winner });

    if (history.length % 20 === 0) {
      const last20 = history.slice(-20);
      const redWins = last20.filter((h) => h.winner === 1).length;
      const blueWins = last20.filter((h) => h.winner === 2).length;
      const draws = last20.filter((h) => h.winner === null).length;
      const decisiveRate = (redWins + blueWins) / 20;
      console.log(`Epizody ${history.length - 19}-${history.length}: czerwoni ${redWins}/20, niebiescy ${blueWins}/20, remis ${draws}/20`);

      targetModel = cloneModel(model);

      if (decisiveRate > bestSuccessRate) {
        bestSuccessRate = decisiveRate;
        saveWeights(model, WEIGHTS_PATH);
        console.log(`  -> nowy rekord (${(decisiveRate * 100).toFixed(0)}% meczów z golem), zapisuję`);
      }
    }
  }

  console.log("Trening 1v1 zakończony.");
  process.exit(0);
}

createEnv1v1("TU_WKLEJ_TOKEN", async (env) => {
  await train(env);
});
