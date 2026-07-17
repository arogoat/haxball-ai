// Wspólna logika używana zarówno przez aktorów (grają), jak i uczącego się (trenuje).
// Trzymana w jednym miejscu, żeby oba procesy zawsze zgadzały się co do kształtu danych.
const tf = require("@tensorflow/tfjs");

const ACTIONS = [
  [0, -1], [1, -1], [1, 0], [1, 1],
  [0, 1], [-1, 1], [-1, 0], [-1, -1]
];

// Bramka przeciwnika (bot gra w drużynie 1, atakuje bramkę po prawej: x=370, y od -64 do 64)
const GOAL_X = 370;
const GOAL_Y = 0;
const GOAL_HALF_HEIGHT = 64;

const FEATURE_COUNT = 9;

// Dystans do NAJBLIŻSZEGO punktu linii bramki (a nie do jej środka) - jeśli piłka
// jest już w zakresie wysokości bramki, liczy się tylko zbliżanie w x. Inaczej
// strzał lekko obok środka, ale wciąż trafiający w słupek/ścianę, dostawałby
// prawie tyle samo nagrody co strzał faktycznie trafiający w bramkę.
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
  // prędkość piłki - efekt ostatniego uderzenia, żeby bot widział co się właśnie stało z piłką
  const ballSpeedX = state.ballSpeedX / 5;
  const ballSpeedY = state.ballSpeedY / 5;
  return [dx, dy, dist, botSpeedX, botSpeedY, ballToGoalX, ballToGoalY, ballSpeedX, ballSpeedY];
}

function computeReward(prevState, newState) {
  const prevDist = Math.hypot(prevState.ballX - prevState.botX, prevState.ballY - prevState.botY);
  const newDist = Math.hypot(newState.ballX - newState.botX, newState.ballY - newState.botY);
  const touchDistance = newState.ballRadius + newState.botRadius + 3;
  const justTouched = newDist < touchDistance && prevDist >= touchDistance;

  let reward = -0.5;
  reward += (prevDist - newDist) * 5;

  const prevBallGoalDist = ballDistToGoal(prevState);
  const newBallGoalDist = ballDistToGoal(newState);
  reward += (prevBallGoalDist - newBallGoalDist) * 5;

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

function createModel() {
  const model = tf.sequential();
  // 24 neurony - tyle samo co w starym modelu (dqn-weights.json), żeby dało się
  // przenieść wyuczone wagi warstw ukrytych bez zmian (patrz migrate-weights.js).
  model.add(tf.layers.dense({ units: 24, activation: "relu", inputShape: [FEATURE_COUNT] }));
  model.add(tf.layers.dense({ units: 24, activation: "relu" }));
  model.add(tf.layers.dense({ units: ACTIONS.length, activation: "linear" }));
  model.compile({ optimizer: tf.train.adam(0.001), loss: "meanSquaredError" });
  return model;
}

module.exports = {
  ACTIONS,
  GOAL_X,
  GOAL_Y,
  FEATURE_COUNT,
  ballDistToGoal,
  getFeatures,
  computeReward,
  createModel,
};
