// AKTOR 1v1: łączy się z jednym pokojem (2 boty: czerwony + niebieski), gra obiema
// stronami, wysyła doświadczenie OBU perspektyw do learnera. Lokalna kopia modelu
// tylko do wyboru akcji (predict) - nigdy nie trenuje sama.
const { createEnv1v1 } = require("./env-1v1.js");
const tf = require("@tensorflow/tfjs");
const { ACTIONS, getFeatures, computeReward, createModel } = require("./dqn-common-1v1.js");

const token = process.argv[2];
const actorIndex = process.argv[3];
const EPSILON_DECAY_EPISODES = Number(process.argv[4]) || 500;
const START_EPISODE = Number(process.argv[5]) || 0;

const MAX_STEPS = 900; // ~15s - starcia 1v1 o piłkę potrzebują więcej czasu niż puste boisko
const EPSILON_MIN = 0.05;
const ACTION_REPEAT = 5;

let model = createModel();
let episodeCount = START_EPISODE;
let epsilon = Math.max(EPSILON_MIN, 0.5 - 0.45 * Math.min(1, episodeCount / EPSILON_DECAY_EPISODES));

process.on("message", (msg) => {
  if (msg.type === "weights") {
    model.setWeights(msg.weights.map((w) => tf.tensor(w)));
  } else if (msg.type === "stop") {
    console.log("otrzymano sygnał stop, kończę");
    process.exit(0);
  }
});

function chooseAction(features) {
  if (Math.random() < epsilon) return Math.floor(Math.random() * ACTIONS.length);
  const qValues = tf.tidy(() => model.predict(tf.tensor2d([features])).dataSync());
  let best = 0;
  for (let a = 1; a < qValues.length; a++) if (qValues[a] > qValues[best]) best = a;
  return best;
}

createEnv1v1(token, async (env) => {
  console.log(`aktor 1v1 ${actorIndex} wystartował (epizod startowy: ${START_EPISODE}, epsilon: ${epsilon.toFixed(2)})`);

  // eslint-disable-next-line no-constant-condition
  while (true) {
    let state = env.reset();
    let redFeatures = getFeatures(state, 1);
    let blueFeatures = getFeatures(state, 2);
    let redAction = chooseAction(redFeatures);
    let blueAction = chooseAction(blueFeatures);
    let winner = null;

    for (let step = 0; step < MAX_STEPS; step++) {
      if (step % ACTION_REPEAT === 0) {
        redAction = chooseAction(redFeatures);
        blueAction = chooseAction(blueFeatures);
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

      // obie perspektywy trafiają do wspólnego bufora learnera - stąd "uniwersalność" modelu
      process.send({
        type: "experience",
        data: { features: redFeatures, action: redAction, reward: redResult.reward, nextFeatures: newRedFeatures, done: redResult.done },
      });
      process.send({
        type: "experience",
        data: { features: blueFeatures, action: blueAction, reward: blueResult.reward, nextFeatures: newBlueFeatures, done: blueResult.done },
      });

      state = newState;
      redFeatures = newRedFeatures;
      blueFeatures = newBlueFeatures;

      if (newState.scoredBy) { winner = newState.scoredBy; break; }
    }

    episodeCount++;
    epsilon = Math.max(EPSILON_MIN, 0.5 - 0.45 * Math.min(1, episodeCount / EPSILON_DECAY_EPISODES));
    process.send({ type: "episode_done", winner });
  }
});
