// AKTOR: łączy się z jednym pokojem HaxBall, gra, i wysyła zdobyte doświadczenie
// do procesu uczącego się (learner.js) przez IPC. Nie trenuje sam - tylko przewiduje
// akcje lokalną kopią sieci, którą co jakiś czas aktualizuje wagami przysłanymi z learnera.
const { createEnv } = require("./env.js");
const tf = require("@tensorflow/tfjs");
const { ACTIONS, getFeatures, computeReward, createModel } = require("./dqn-common.js");

const token = process.argv[2];
const actorIndex = process.argv[3];
// Ile WŁASNYCH epizodów ten aktor odegra, zanim epsilon ma dojść do minimum -
// czyli globalny budżet epizodów podzielony przez liczbę aktorów, bo trenujemy
// współbieżnie (nie każdy aktor osobno robi pełne 500).
const EPSILON_DECAY_EPISODES = Number(process.argv[4]) || 500;
// Ile epizodów ten aktor już rozegrał w POPRZEDNICH uruchomieniach learner.js -
// żeby epsilon kontynuował zanikanie po restarcie, zamiast wracać do 0.5 za każdym razem.
const START_EPISODE = Number(process.argv[5]) || 0;

const MAX_STEPS = 600; // ~10s zamiast ~6.7s - więcej czasu na wahające się/niepewne podejścia
const EPSILON_MIN = 0.05;
// Decyzja o kierunku ruchu podejmowana co tyle klatek (nie co pojedynczą) - zapobiega
// "drganiu w miejscu", gdy dwie przeciwne akcje mają prawie równą wartość i bot co
// klatkę zmienia zdanie zamiast się zdecydować. Wymusza też dłuższe, płynniejsze ruchy.
const ACTION_REPEAT = 5;

let model = createModel(); // lokalna kopia tylko do wyboru akcji (predict), nigdy nie trenowana tutaj
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

createEnv(token, async (env) => {
  console.log(`aktor ${actorIndex} wystartował (epizod startowy: ${START_EPISODE}, epsilon: ${epsilon.toFixed(2)})`);

  // eslint-disable-next-line no-constant-condition
  while (true) {
    let state = env.reset();
    let features = getFeatures(state);
    let success = false;
    let action = chooseAction(features);

    for (let step = 0; step < MAX_STEPS; step++) {
      if (step % ACTION_REPEAT === 0) {
        action = chooseAction(features);
      }
      const [dirX, dirY] = ACTIONS[action];

      const currentDist = Math.hypot(state.ballX - state.botX, state.ballY - state.botY);
      const kick = currentDist < (state.ballRadius + state.botRadius + 5);

      const newState = await env.step(dirX, dirY, kick);
      const newFeatures = getFeatures(newState);
      const { reward, done } = computeReward(state, newState);

      process.send({
        type: "experience",
        data: { features, action, reward, nextFeatures: newFeatures, done },
      });

      state = newState;
      features = newFeatures;

      if (done) { success = true; break; }
    }

    episodeCount++;
    epsilon = Math.max(EPSILON_MIN, 0.5 - 0.45 * Math.min(1, episodeCount / EPSILON_DECAY_EPISODES));
    process.send({ type: "episode_done", success });
  }
});
