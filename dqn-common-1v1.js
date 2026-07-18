// Wspólna logika dla trybu 1v1. Dane wejściowe liczone są WZGLĘDEM DRUŻYNY bota
// (flip współrzędnej x dla drużyny 2), więc ten sam kod/model działa identycznie
// niezależnie po której stronie boiska gra - "moja bramka" i "bramka przeciwnika"
// są zawsze na tych samych względnych pozycjach. To też grunt pod ewentualny
// self-play (jeden model po obu stronach) w przyszłości.
const tf = require("@tensorflow/tfjs");

const ACTIONS = [
  [0, -1], [1, -1], [1, 0], [1, 1],
  [0, 1], [-1, 1], [-1, 0], [-1, -1]
];

const GOAL_X = 370;
const GOAL_HALF_HEIGHT = 64;

// Nagrody licza sie RAZ NA DECYZJE (okno kilku tickow), nie raz na tick - patrz
// actor-1v1.js. Kara za czas jest teraz per-decyzja, wiec calkowita kara za caly
// mecz do limitu krokow jest wyraznie mniejsza niz utrata gola (patrz TIMEOUT_PENALTY),
// zamiast (jak wczesniej) niemal rownowazna stracie punktu.
const STEP_PENALTY = -0.5;
const TOUCH_REWARD = 15;
const GOAL_REWARD = 500;
const TIMEOUT_PENALTY = -30;

// 9 pierwszych cech ma DOKŁADNIE to samo znaczenie i kolejność co w modelu 1-botowym
// (dqn-common.js) - żeby dało się przenieść już wyuczone wagi. 5 nowych na końcu
// (przeciwnik) zostaje losowo zainicjalizowanych przy migracji.
const FEATURE_COUNT = 14;

function getOwnAndOpponent(state, team) {
  if (team === 1) {
    return {
      selfX: state.redX, selfY: state.redY,
      selfSpeedX: state.redSpeedX, selfSpeedY: state.redSpeedY,
      selfRadius: state.redRadius,
      oppX: state.blueX, oppY: state.blueY,
      oppSpeedX: state.blueSpeedX, oppSpeedY: state.blueSpeedY,
      oppRadius: state.blueRadius,
    };
  }
  return {
    selfX: state.blueX, selfY: state.blueY,
    selfSpeedX: state.blueSpeedX, selfSpeedY: state.blueSpeedY,
    selfRadius: state.blueRadius,
    oppX: state.redX, oppY: state.redY,
    oppSpeedX: state.redSpeedX, oppSpeedY: state.redSpeedY,
    oppRadius: state.redRadius,
  };
}

function getFeatures(state, team) {
  const flip = team === 2 ? -1 : 1;
  const own = getOwnAndOpponent(state, team);

  const ballX = state.ballX * flip;
  const ballY = state.ballY;
  const selfX = own.selfX * flip;
  const selfY = own.selfY;
  const oppX = own.oppX * flip;
  const oppY = own.oppY;

  const dx = (ballX - selfX) / 400;
  const dy = (ballY - selfY) / 400;
  const dist = Math.hypot(ballX - selfX, ballY - selfY) / 400;
  const selfSpeedX = (own.selfSpeedX * flip) / 5;
  const selfSpeedY = own.selfSpeedY / 5;
  const ballToGoalX = (GOAL_X - ballX) / 800;
  const ballToGoalY = (0 - ballY) / 200;
  const ballSpeedX = (state.ballSpeedX * flip) / 5;
  const ballSpeedY = state.ballSpeedY / 5;

  const oppDx = (oppX - selfX) / 400;
  const oppDy = (oppY - selfY) / 400;
  const oppDist = Math.hypot(oppX - selfX, oppY - selfY) / 400;
  const oppSpeedX = (own.oppSpeedX * flip) / 5;
  const oppSpeedY = own.oppSpeedY / 5;

  return [
    dx, dy, dist, selfSpeedX, selfSpeedY,
    ballToGoalX, ballToGoalY, ballSpeedX, ballSpeedY,
    oppDx, oppDy, oppDist, oppSpeedX, oppSpeedY,
  ];
}

function ballDistToGoal(ballX, ballY, team) {
  const targetX = team === 1 ? GOAL_X : -GOAL_X;
  const targetY = Math.max(-GOAL_HALF_HEIGHT, Math.min(GOAL_HALF_HEIGHT, ballY));
  return Math.hypot(targetX - ballX, targetY - ballY);
}

// prevState/newState to teraz stan PRZED i PO calym oknie decyzji (kilka
// tickow), nie pojedynczy tick - patrz actor-1v1.js. touchedDuringWindow -
// czy bot dotknal pilki w KTORYMKOLWIEK ticku tego okna (liczone przez
// wywolujacego, bo tylko on widzi kazdy pojedynczy tick). isTimeout=true
// oznacza, ze to byla ostatnia decyzja przed limitem krokow - wtedy epizod
// KONCZY SIE tu (done=true), zeby siec nie uczyla sie fantomowej "dalszej
// gry" z nastepnego, w rzeczywistosci juz nowego epizodu.
function computeReward(prevState, newState, team, isTimeout, touchedDuringWindow) {
  const prevOwn = getOwnAndOpponent(prevState, team);
  const newOwn = getOwnAndOpponent(newState, team);

  const prevDist = Math.hypot(prevState.ballX - prevOwn.selfX, prevState.ballY - prevOwn.selfY);
  const newDist = Math.hypot(newState.ballX - newOwn.selfX, newState.ballY - newOwn.selfY);

  let reward = STEP_PENALTY;
  reward += (prevDist - newDist) * 5;

  const prevBallGoalDist = ballDistToGoal(prevState.ballX, prevState.ballY, team);
  const newBallGoalDist = ballDistToGoal(newState.ballX, newState.ballY, team);
  reward += (prevBallGoalDist - newBallGoalDist) * 5;

  if (touchedDuringWindow) reward += TOUCH_REWARD;

  if (newState.scoredBy === team) {
    reward += GOAL_REWARD;
    return { reward, done: true, touched: touchedDuringWindow };
  }
  if (newState.scoredBy !== null && newState.scoredBy !== team) {
    reward -= GOAL_REWARD;
    return { reward, done: true, touched: touchedDuringWindow };
  }

  if (isTimeout) {
    reward += TIMEOUT_PENALTY;
    return { reward, done: true, touched: touchedDuringWindow };
  }

  return { reward, done: false, touched: touchedDuringWindow };
}

function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 24, activation: "relu", inputShape: [FEATURE_COUNT] }));
  model.add(tf.layers.dense({ units: 24, activation: "relu" }));
  model.add(tf.layers.dense({ units: ACTIONS.length, activation: "linear" }));
  model.compile({ optimizer: tf.train.adam(0.001), loss: "meanSquaredError" });
  return model;
}

module.exports = {
  ACTIONS,
  FEATURE_COUNT,
  getFeatures,
  computeReward,
  createModel,
  STEP_PENALTY,
  TOUCH_REWARD,
  GOAL_REWARD,
  TIMEOUT_PENALTY,
};
