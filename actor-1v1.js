// AKTOR 1v1: łączy się z jednym pokojem (2 boty: czerwony + niebieski), gra obiema
// stronami, wysyła doświadczenie OBU perspektyw do learnera. Lokalna kopia modelu
// tylko do wyboru akcji (predict) - nigdy nie trenuje sama.
//
// WAŻNE: doświadczenie wysyłane jest RAZ NA DECYZJĘ (okno ACTION_REPEAT tickow),
// nie raz na tick. Wczesniej, przy 60 tickach/s i gamma=0.95, gol oddalony o samo
// 2 sekundy (120 tickow) mial po zdyskontowaniu wartosc ~0.001 - siec w praktyce
// nie widziala celu gry, tylko shaping. Agregacja do poziomu decyzji (5 tickow =
// ~83ms) razem z podniesieniem gamma w learnerze naprawia horyzont czasowy.
const { createEnv1v1 } = require("./env-1v1.js");
const tf = require("@tensorflow/tfjs");
const { ACTIONS, getFeatures, computeReward, createModel } = require("./dqn-common-1v1.js");

const token = process.argv[2];
const actorIndex = process.argv[3];
const EPSILON_DECAY_EPISODES = Number(process.argv[4]) || 3000;
const START_EPISODE = Number(process.argv[5]) || 0;
const EPSILON_MIN = Number(process.argv[6]) || 0.1;

const MAX_STEPS = 900; // ~15s - starcia 1v1 o piłkę potrzebują więcej czasu niż puste boisko
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

function isTouching(state, x, y, radiusField) {
  const dist = Math.hypot(state.ballX - x, state.ballY - y);
  return dist < (state.ballRadius + state[radiusField] + 3);
}

createEnv1v1(token, async (env) => {
  console.log(`aktor 1v1 ${actorIndex} wystartował (epizod startowy: ${START_EPISODE}, epsilon: ${epsilon.toFixed(2)}, decay: ${EPSILON_DECAY_EPISODES} epizodow, min: ${EPSILON_MIN})`);

  // eslint-disable-next-line no-constant-condition
  while (true) {
    let state = env.reset();
    let redFeatures = getFeatures(state, 1);
    let blueFeatures = getFeatures(state, 2);
    let winner = null;

    // Telemetria calego epizodu - zeby dalo sie wykryc "zamrozona" polityke
    // (boty stojace w miejscu / ignorujace pilke) z samych liczb, bez ogladania
    // meczu w przegladarce.
    let totalTicks = 0;
    let redDistSum = 0, blueDistSum = 0;
    let redSpeedSum = 0, blueSpeedSum = 0;
    let redTouches = 0, blueTouches = 0;
    let redRewardSum = 0, blueRewardSum = 0;
    let decisions = 0;
    const redActionCounts = new Array(ACTIONS.length).fill(0);
    const blueActionCounts = new Array(ACTIONS.length).fill(0);

    let step = 0;
    while (step < MAX_STEPS) {
      const redAction = chooseAction(redFeatures);
      const blueAction = chooseAction(blueFeatures);
      redActionCounts[redAction]++;
      blueActionCounts[blueAction]++;
      const [redDirX, redDirY] = ACTIONS[redAction];
      const [blueDirX, blueDirY] = ACTIONS[blueAction];

      const windowStartState = state;
      let scored = null;
      let redTouchedWindow = false;
      let blueTouchedWindow = false;

      for (let i = 0; i < ACTION_REPEAT && step < MAX_STEPS; i++, step++) {
        const redDist = Math.hypot(state.ballX - state.redX, state.ballY - state.redY);
        const blueDist = Math.hypot(state.ballX - state.blueX, state.ballY - state.blueY);
        const redKick = redDist < (state.ballRadius + state.redRadius + 5);
        const blueKick = blueDist < (state.ballRadius + state.blueRadius + 5);

        state = await env.step(redDirX, redDirY, redKick, blueDirX, blueDirY, blueKick);
        totalTicks++;

        redDistSum += Math.hypot(state.ballX - state.redX, state.ballY - state.redY);
        blueDistSum += Math.hypot(state.ballX - state.blueX, state.ballY - state.blueY);
        redSpeedSum += Math.hypot(state.redSpeedX, state.redSpeedY);
        blueSpeedSum += Math.hypot(state.blueSpeedX, state.blueSpeedY);
        if (isTouching(state, state.redX, state.redY, "redRadius")) redTouchedWindow = true;
        if (isTouching(state, state.blueX, state.blueY, "blueRadius")) blueTouchedWindow = true;

        if (state.scoredBy) { scored = state.scoredBy; break; }
      }

      decisions++;
      const isTimeout = !scored && step >= MAX_STEPS;
      const newRedFeatures = getFeatures(state, 1);
      const newBlueFeatures = getFeatures(state, 2);
      const redResult = computeReward(windowStartState, state, 1, isTimeout, redTouchedWindow);
      const blueResult = computeReward(windowStartState, state, 2, isTimeout, blueTouchedWindow);
      if (redResult.touched) redTouches++;
      if (blueResult.touched) blueTouches++;
      redRewardSum += redResult.reward;
      blueRewardSum += blueResult.reward;

      // obie perspektywy trafiają do wspólnego bufora learnera - stąd "uniwersalność" modelu
      process.send({
        type: "experience",
        data: { features: redFeatures, action: redAction, reward: redResult.reward, nextFeatures: newRedFeatures, done: redResult.done },
      });
      process.send({
        type: "experience",
        data: { features: blueFeatures, action: blueAction, reward: blueResult.reward, nextFeatures: newBlueFeatures, done: blueResult.done },
      });

      redFeatures = newRedFeatures;
      blueFeatures = newBlueFeatures;

      if (scored) { winner = scored; break; }
      if (isTimeout) break;
    }

    episodeCount++;
    epsilon = Math.max(EPSILON_MIN, 0.5 - 0.45 * Math.min(1, episodeCount / EPSILON_DECAY_EPISODES));

    process.send({
      type: "episode_done",
      winner,
      telemetry: {
        ticks: totalTicks,
        decisions,
        epsilon,
        redAvgDist: totalTicks ? redDistSum / totalTicks : 0,
        blueAvgDist: totalTicks ? blueDistSum / totalTicks : 0,
        redAvgSpeed: totalTicks ? redSpeedSum / totalTicks : 0,
        blueAvgSpeed: totalTicks ? blueSpeedSum / totalTicks : 0,
        redTouches,
        blueTouches,
        redRewardSum,
        blueRewardSum,
        redActionCounts,
        blueActionCounts,
      },
    });
  }
});
