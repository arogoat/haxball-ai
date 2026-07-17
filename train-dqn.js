const { createEnv } = require("./env.js");
const tf = require("@tensorflow/tfjs");
const fs = require("fs");

const ACTIONS = [
  [0, -1], [1, -1], [1, 0], [1, 1],
  [0, 1], [-1, 1], [-1, 0], [-1, -1]
];

// Bramka przeciwnika (bot gra w drużynie 1, atakuje bramkę po prawej: x=370, y od -64 do 64)
const GOAL_X = 370;
const GOAL_Y = 0;
const GOAL_HALF_HEIGHT = 64;

function ballDistToGoal(state) {
  const targetY = Math.max(-GOAL_HALF_HEIGHT, Math.min(GOAL_HALF_HEIGHT, state.ballY));
  return Math.hypot(GOAL_X - state.ballX, targetY - state.ballY);
}

function getFeatures(state) {
  const dx = (state.ballX - state.botX) / 400;
  const dy = (state.ballY - state.botY) / 400;
  const dist = Math.hypot(state.ballX - state.botX, state.ballY - state.botY) / 400;
  const botSpeedX = state.botSpeedX / 5;
  const botSpeedY = state.botSpeedY / 5;
  const ballToGoalX = (GOAL_X - state.ballX) / 800;
  const ballToGoalY = (GOAL_Y - state.ballY) / 200;
  const ballSpeedX = state.ballSpeedX / 5;
  const ballSpeedY = state.ballSpeedY / 5;
  return [dx, dy, dist, botSpeedX, botSpeedY, ballToGoalX, ballToGoalY, ballSpeedX, ballSpeedY];
}

const FEATURE_COUNT = 9;

function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 24, activation: "relu", inputShape: [FEATURE_COUNT] }));
  model.add(tf.layers.dense({ units: 24, activation: "relu" }));
  model.add(tf.layers.dense({ units: ACTIONS.length, activation: "linear" }));
  model.compile({ optimizer: tf.train.adam(0.001), loss: "meanSquaredError" });
  return model;
}

function cloneModel(model) {
  const clone = createModel();
  clone.setWeights(model.getWeights().map(w => w.clone()));
  return clone;
}

function chooseAction(model, features, epsilon) {
  if (Math.random() < epsilon) return Math.floor(Math.random() * ACTIONS.length);
  const qValues = tf.tidy(() => model.predict(tf.tensor2d([features])).dataSync());
  let best = 0;
  for (let a = 1; a < qValues.length; a++) if (qValues[a] > qValues[best]) best = a;
  return best;
}

const replayBuffer = [];
const MAX_BUFFER = 5000;
function remember(features, action, reward, nextFeatures, done) {
  replayBuffer.push({ features, action, reward, nextFeatures, done });
  if (replayBuffer.length > MAX_BUFFER) replayBuffer.shift();
}

async function trainOnBatch(model, targetModel, batchSize, gamma) {
  if (replayBuffer.length < batchSize) return;

  const batch = [];
  for (let i = 0; i < batchSize; i++) {
    batch.push(replayBuffer[Math.floor(Math.random() * replayBuffer.length)]);
  }

  const states = batch.map((b) => b.features);
  const nextStates = batch.map((b) => b.nextFeatures);

  const qCurrent = model.predict(tf.tensor2d(states)).arraySync();
  const qNext = targetModel.predict(tf.tensor2d(nextStates)).arraySync();

  const targets = qCurrent.map((qValues, i) => {
    const { action, reward, done } = batch[i];
    const target = qValues.slice();
    const maxNextQ = Math.max(...qNext[i]);
    target[action] = done ? reward : reward + gamma * maxNextQ;
    return target;
  });

  await model.fit(tf.tensor2d(states), tf.tensor2d(targets), { epochs: 1, verbose: 0 });
}

function computeReward(prevState, newState) {
  const prevDist = Math.hypot(prevState.ballX - prevState.botX, prevState.ballY - prevState.botY);
  const newDist = Math.hypot(newState.ballX - newState.botX, newState.ballY - newState.botY);
  const touchDistance = newState.ballRadius + newState.botRadius + 3;
  // liczymy "dotknięcie" tylko przy przejściu z dala->blisko, żeby nie zbierać nagrody co klatkę stojąc przy piłce
  const justTouched = newDist < touchDistance && prevDist >= touchDistance;

  let reward = -0.5;
  reward += (prevDist - newDist) * 5; // dojście do piłki

  const prevBallGoalDist = ballDistToGoal(prevState);
  const newBallGoalDist = ballDistToGoal(newState);
  reward += (prevBallGoalDist - newBallGoalDist) * 5; // pchanie piłki w stronę bramki

  if (justTouched) reward += 20;

  if (newState.goalScored) {
    reward += 500;
    return { reward, done: true };
  }

  if (newState.ownGoal) {
    reward -= 500;
    return { reward, done: true };
  }

  return { reward, done: false };
}

function saveWeights(model, path) {
  const weights = model.getWeights().map((w) => w.arraySync());
  fs.writeFileSync(path, JSON.stringify(weights));
}

function loadWeightsIfExist(model, path) {
  if (!fs.existsSync(path)) {
    console.log("Brak zapisanych wag - zaczynam od losowej sieci.");
    return;
  }
  try {
    const saved = JSON.parse(fs.readFileSync(path, "utf8"));
    const tensors = saved.map((w) => tf.tensor(w));
    model.setWeights(tensors);
    console.log(`Wczytano wagi z ${path} - kontynuuję trening zamiast zaczynać od zera.`);
  } catch (err) {
    console.log("Zapisane wagi nie pasują do obecnej architektury sieci (zmieniła się liczba featurów) - zaczynam od losowej sieci.");
  }
}

const TOTAL_EPISODES = 500;
const MAX_STEPS = 600; // strzelenie gola zajmuje więcej kroków niż samo dotknięcie piłki
const ACTION_REPEAT = 5; // decyzja o kierunku co 5 klatek, zapobiega drganiu w miejscu
const GAMMA = 0.95; // wyżej niż wcześniej, bo nagroda za gola przychodzi później
const BATCH_SIZE = 32;

// Argumenty z linii komend (opcjonalne) - potrzebne, żeby ten sam plik dało się
// odpalić wiele razy naraz jako osobne procesy, każdy z innym tokenem.
// Użycie pojedyncze (jak dotychczas):   node train-dqn.js
// Użycie równoległe (z orkiestratora):  node train-dqn.js <token> <plikWag> <plikWyniku>
const TOKEN = process.argv[2] || "TU_WKLEJ_SWIEZY_TOKEN";
const WEIGHTS_PATH = process.argv[3] || "dqn-weights-goal.json";
const RESULT_PATH = process.argv[4] || null; // jeśli podane, na końcu zapisujemy tu najlepszy wynik

async function train(env, model) {
  const rewardHistory = [];
  let bestSuccessRate = -1;
  let targetModel = cloneModel(model);

  for (let ep = 0; ep < TOTAL_EPISODES; ep++) {
    const progress = ep / TOTAL_EPISODES;
    const epsilon = Math.max(0.05, 0.5 - 0.45 * progress);

    let state = env.reset();
    let features = getFeatures(state);
    let totalReward = 0;
    let success = false;
    let action = chooseAction(model, features, epsilon);

    for (let step = 0; step < MAX_STEPS; step++) {
      if (step % ACTION_REPEAT === 0) {
        action = chooseAction(model, features, epsilon);
      }
      const [dirX, dirY] = ACTIONS[action];

      const currentDist = Math.hypot(state.ballX - state.botX, state.ballY - state.botY);
      const kick = currentDist < (state.ballRadius + state.botRadius + 5);

      const newState = await env.step(dirX, dirY, kick);
      const newFeatures = getFeatures(newState);
      const { reward, done } = computeReward(state, newState);

      remember(features, action, reward, newFeatures, done);
      await trainOnBatch(model, targetModel, BATCH_SIZE, GAMMA);

      totalReward += reward;
      state = newState;
      features = newFeatures;
      if (done) { success = true; break; }
    }

    rewardHistory.push({ totalReward, success });
    if (rewardHistory.length % 20 === 0) {
      const last20 = rewardHistory.slice(-20);
      const avgReward = last20.reduce((s, r) => s + r.totalReward, 0) / last20.length;
      const successRate = last20.filter((r) => r.success).length / last20.length;
      console.log(`Epizody ${rewardHistory.length - 19}-${rewardHistory.length}: średnia = ${avgReward.toFixed(1)}, skuteczność = ${(successRate * 100).toFixed(0)}%`);

      targetModel = cloneModel(model);

      if (successRate > bestSuccessRate) {
        bestSuccessRate = successRate;
        saveWeights(model, WEIGHTS_PATH);
        console.log(`  -> nowy rekord (${(bestSuccessRate * 100).toFixed(0)}%), zapisuję wagi na dysk`);
      }
    }
  }

  console.log(`Trening zakończony. Najlepszy wynik: ${(bestSuccessRate * 100).toFixed(0)}%`);

  if (RESULT_PATH) {
    fs.writeFileSync(RESULT_PATH, JSON.stringify({ bestSuccessRate, weightsPath: WEIGHTS_PATH }));
  }

  process.exit(0);
}

createEnv(TOKEN, async (env) => {
  const model = createModel();
  loadWeightsIfExist(model, WEIGHTS_PATH);
  await train(env, model);
});
