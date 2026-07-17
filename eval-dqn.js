const { createEnv } = require("./env.js");
const tf = require("@tensorflow/tfjs");
const fs = require("fs");

const ACTIONS = [
  [0, -1], [1, -1], [1, 0], [1, 1],
  [0, 1], [-1, 1], [-1, 0], [-1, -1]
];

const GOAL_X = 370;
const GOAL_Y = 0;

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

function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 24, activation: "relu", inputShape: [9] }));
  model.add(tf.layers.dense({ units: 24, activation: "relu" }));
  model.add(tf.layers.dense({ units: ACTIONS.length, activation: "linear" }));
  model.compile({ optimizer: tf.train.adam(0.001), loss: "meanSquaredError" });
  return model;
}

function loadWeights(model, path) {
  const weightsData = JSON.parse(fs.readFileSync(path, "utf8"));
  const currentWeights = model.getWeights();
  const tensors = weightsData.map((w, i) => tf.tensor(w, currentWeights[i].shape));
  model.setWeights(tensors);
}

function chooseAction(model, features, epsilon) {
  if (Math.random() < epsilon) return Math.floor(Math.random() * ACTIONS.length);
  const qValues = tf.tidy(() => model.predict(tf.tensor2d([features])).dataSync());
  let best = 0;
  for (let a = 1; a < qValues.length; a++) if (qValues[a] > qValues[best]) best = a;
  return best;
}

const MAX_STEPS = 600;

async function evaluate(env, model, episodes, epsilon) {
  let successes = 0;
  for (let ep = 0; ep < episodes; ep++) {
    let state = env.reset();
    let features = getFeatures(state);
    let success = false;
    for (let step = 0; step < MAX_STEPS; step++) {
      const action = chooseAction(model, features, epsilon);
      const [dirX, dirY] = ACTIONS[action];
      const currentDist = Math.hypot(state.ballX - state.botX, state.ballY - state.botY);
      const kick = currentDist < (state.ballRadius + state.botRadius + 5);
      const newState = await env.step(dirX, dirY, kick);
      state = newState;
      features = getFeatures(newState);
      if (newState.goalScored) { success = true; break; }
    }
    if (success) successes++;
  }
  console.log(`Ewaluacja (epsilon=${epsilon}): ${successes}/${episodes} goli (${(100 * successes / episodes).toFixed(0)}%)`);
}

createEnv("TU_WKLEJ_SWIEZY_TOKEN", async (env) => {
  const model = createModel();
  loadWeights(model, "dqn-weights-goal.json");
  await evaluate(env, model, 30, 0);
});