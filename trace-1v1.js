// Diagnostyka zachowania: rozgrywa kilka czystych meczow (epsilon=0) i zapisuje
// do CSV pozycje pilki/botow, wybrane akcje i max Q co kilka krokow - zeby dalo
// sie zobaczyc CO DOKLADNIE robi model, a nie tylko czy mecz wygral/przegral.
const { createEnv1v1 } = require("./env-1v1.js");
const tf = require("@tensorflow/tfjs");
const { ACTIONS, getFeatures, createModel } = require("./dqn-common-1v1.js");
const fs = require("fs");

const WEIGHTS_PATH = "dqn-weights-1v1.json";
const MAX_STEPS = 900;
const EPISODES = 3;
const SAMPLE_EVERY = 5;
const OUT_PATH = "trace-1v1.csv";

const ACTION_NAMES = ["gora", "gora-prawo", "prawo", "dol-prawo", "dol", "dol-lewo", "lewo", "gora-lewo"];

function loadWeights(model, path) {
  const weightsData = JSON.parse(fs.readFileSync(path, "utf8"));
  const currentWeights = model.getWeights();
  const tensors = weightsData.map((w, i) => tf.tensor(w, currentWeights[i].shape));
  model.setWeights(tensors);
}

function qValuesFor(model, features) {
  return tf.tidy(() => Array.from(model.predict(tf.tensor2d([features])).dataSync()));
}

function bestAction(qValues) {
  let best = 0;
  for (let a = 1; a < qValues.length; a++) if (qValues[a] > qValues[best]) best = a;
  return best;
}

async function trace(env) {
  const model = createModel();
  loadWeights(model, WEIGHTS_PATH);

  const rows = ["episode,step,ballX,ballY,redX,redY,redAction,redQmax,blueX,blueY,blueAction,blueQmax,scoredBy"];

  for (let ep = 0; ep < EPISODES; ep++) {
    let state = env.reset();
    for (let step = 0; step < MAX_STEPS; step++) {
      const redFeatures = getFeatures(state, 1);
      const blueFeatures = getFeatures(state, 2);
      const redQ = qValuesFor(model, redFeatures);
      const blueQ = qValuesFor(model, blueFeatures);
      const redAction = bestAction(redQ);
      const blueAction = bestAction(blueQ);
      const [redDirX, redDirY] = ACTIONS[redAction];
      const [blueDirX, blueDirY] = ACTIONS[blueAction];

      const redDist = Math.hypot(state.ballX - state.redX, state.ballY - state.redY);
      const blueDist = Math.hypot(state.ballX - state.blueX, state.ballY - state.blueY);
      const redKick = redDist < (state.ballRadius + state.redRadius + 5);
      const blueKick = blueDist < (state.ballRadius + state.blueRadius + 5);

      if (step % SAMPLE_EVERY === 0) {
        rows.push([
          ep, step,
          state.ballX.toFixed(1), state.ballY.toFixed(1),
          state.redX.toFixed(1), state.redY.toFixed(1),
          ACTION_NAMES[redAction], Math.max(...redQ).toFixed(2),
          state.blueX.toFixed(1), state.blueY.toFixed(1),
          ACTION_NAMES[blueAction], Math.max(...blueQ).toFixed(2),
          "",
        ].join(","));
      }

      state = await env.step(redDirX, redDirY, redKick, blueDirX, blueDirY, blueKick);

      if (state.scoredBy) {
        rows.push([
          ep, step + 1,
          state.ballX.toFixed(1), state.ballY.toFixed(1),
          state.redX.toFixed(1), state.redY.toFixed(1),
          "", "",
          state.blueX.toFixed(1), state.blueY.toFixed(1),
          "", "",
          state.scoredBy,
        ].join(","));
        break;
      }
    }
  }

  fs.writeFileSync(OUT_PATH, rows.join("\n"));
  console.log(`Zapisano ${rows.length - 1} wierszy do ${OUT_PATH}`);
}

const token = process.argv[2];
if (!token) {
  console.error("Uzycie: node trace-1v1.js <swiezy_token>");
  process.exit(1);
}

createEnv1v1(token, async (env) => {
  await trace(env);
  process.exit(0);
});
