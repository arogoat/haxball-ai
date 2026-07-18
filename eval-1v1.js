// Czysta ewaluacja wytrenowanego modelu 1v1: obaj boci (czerwony i niebieski)
// gra TYM SAMYM wyuczonym modelem z epsilon=0 (zero losowosci), zeby zobaczyc
// realny, aktualny poziom gry bez szumu eksploracji, ktory zaniza statystyki
// widoczne w trakcie samego treningu.
const { createEnv1v1 } = require("./env-1v1.js");
const tf = require("@tensorflow/tfjs");
const { ACTIONS, getFeatures, createModel } = require("./dqn-common-1v1.js");
const fs = require("fs");

const MAX_STEPS = 900;
const WEIGHTS_PATH = "dqn-weights-1v1.json";
const EPISODES = 50;

function loadWeights(model, path) {
  const weightsData = JSON.parse(fs.readFileSync(path, "utf8"));
  const currentWeights = model.getWeights();
  const tensors = weightsData.map((w, i) => tf.tensor(w, currentWeights[i].shape));
  model.setWeights(tensors);
}

function chooseAction(model, features) {
  const qValues = tf.tidy(() => model.predict(tf.tensor2d([features])).dataSync());
  let best = 0;
  for (let a = 1; a < qValues.length; a++) if (qValues[a] > qValues[best]) best = a;
  return best;
}

async function evaluate(env, model, episodes) {
  let redWins = 0, blueWins = 0, draws = 0;
  let totalSteps = 0;

  for (let ep = 0; ep < episodes; ep++) {
    let state = env.reset();
    let winner = null;
    let steps = 0;

    for (let step = 0; step < MAX_STEPS; step++) {
      const redFeatures = getFeatures(state, 1);
      const blueFeatures = getFeatures(state, 2);
      const redAction = chooseAction(model, redFeatures);
      const blueAction = chooseAction(model, blueFeatures);
      const [redDirX, redDirY] = ACTIONS[redAction];
      const [blueDirX, blueDirY] = ACTIONS[blueAction];

      const redDist = Math.hypot(state.ballX - state.redX, state.ballY - state.redY);
      const blueDist = Math.hypot(state.ballX - state.blueX, state.ballY - state.blueY);
      const redKick = redDist < (state.ballRadius + state.redRadius + 5);
      const blueKick = blueDist < (state.ballRadius + state.blueRadius + 5);

      state = await env.step(redDirX, redDirY, redKick, blueDirX, blueDirY, blueKick);
      steps++;

      if (state.scoredBy) { winner = state.scoredBy; break; }
    }

    totalSteps += steps;
    if (winner === 1) redWins++;
    else if (winner === 2) blueWins++;
    else draws++;

    console.log(`Mecz ${ep + 1}/${episodes}: ${winner === 1 ? "czerwoni" : winner === 2 ? "niebiescy" : "remis"} (${steps} krokow)`);
  }

  console.log("");
  console.log(`Wyniki po ${episodes} meczach (epsilon=0, ten sam model po obu stronach):`);
  console.log(`  czerwoni: ${redWins} (${(100 * redWins / episodes).toFixed(0)}%)`);
  console.log(`  niebiescy: ${blueWins} (${(100 * blueWins / episodes).toFixed(0)}%)`);
  console.log(`  remisy: ${draws} (${(100 * draws / episodes).toFixed(0)}%)`);
  console.log(`  sredni czas meczu: ${(totalSteps / episodes).toFixed(0)} krokow (limit ${MAX_STEPS})`);
}

const token = process.argv[2];
if (!token) {
  console.error("Uzycie: node eval-1v1.js <swiezy_token>");
  process.exit(1);
}

createEnv1v1(token, async (env) => {
  const model = createModel();
  loadWeights(model, WEIGHTS_PATH);
  await evaluate(env, model, EPISODES);
  process.exit(0);
});
