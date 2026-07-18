// Środowisko 1v1: hostuje pokój z DWOMA botami (czerwony - drużyna 1, niebieski -
// drużyna 2), zamiast jednym. step() przyjmuje akcje dla obu naraz, readState()
// zwraca stan piłki i obu botów, plus informację który zespół strzelił gola
// (uwzględniając samobóje - jeśli czerwony wbije do własnej bramki, liczy się to
// jako gol dla niebieskiego, zgodnie z normalnymi zasadami).
const { Room, Utils } = require("node-haxball")();

const RED_ID = 1;
const BLUE_ID = 2;
const BALL_SPAWN = { x: 0, y: 0 };
const RED_SPAWN = { x: -277.5, y: 0 };
const BLUE_SPAWN = { x: 277.5, y: 0 };

function createEnv1v1(token, onReady) {
  let room = null;
  let tickWaiters = [];
  let scoredBy = null; // 1 (czerwoni), 2 (niebiescy), albo null

  function readState() {
    const ball = room.getBall();
    const red = room.getPlayerDisc(RED_ID);
    const blue = room.getPlayerDisc(BLUE_ID);
    // W trakcie automatycznego restartu meczu (onGameStop -> startGame) discs
    // graczy bywają chwilowo niedostępne, zanim silnik je odtworzy. Zamiast
    // crashować cały proces aktora (i przez to cały learner przez zerwane IPC),
    // zwracamy null - onGameTick po prostu poczeka na kolejny, poprawny tick.
    if (!ball || !red || !blue) return null;
    const scored = scoredBy;
    scoredBy = null;
    return {
      ballX: ball.pos.x, ballY: ball.pos.y,
      ballSpeedX: ball.speed.x, ballSpeedY: ball.speed.y,
      ballRadius: ball.radius,
      redX: red.pos.x, redY: red.pos.y,
      redSpeedX: red.speed.x, redSpeedY: red.speed.y,
      redRadius: red.radius,
      blueX: blue.pos.x, blueY: blue.pos.y,
      blueSpeedX: blue.speed.x, blueSpeedY: blue.speed.y,
      blueRadius: blue.radius,
      scoredBy: scored,
    };
  }

  // Handler linku podpiety JUZ na poziomie Room.create (nie w srodku onOpen) -
  // zdarzenie onRoomLink potrafi odpalic sie ZANIM kod w onOpen zdazy przypisac
  // room.onRoomLink, i wtedy link przepada bez sladu (dokladnie to obserwowalismy:
  // brak linkow w konsoli). Dodatkowo linki gina w zalewie logow telemetrii,
  // wiec zapisujemy je tez do pliku: cat room-links-1v1.txt
  const onRoomLink = (link) => {
    console.log("Otwórz w przeglądarce, żeby oglądać:", link);
    try {
      require("fs").appendFileSync(
        "room-links-1v1.txt",
        `${new Date().toISOString()} pid=${process.pid} ${link}\n`
      );
    } catch (e) { /* brak zapisu do pliku nie moze psuc treningu */ }
  };

  Room.create({
    name: "haxball-ai-1v1",
    noPlayer: true,
    maxPlayerCount: 10,
    token
  }, {
    storage: { player_name: "host" },
    onRoomLink,
    onOpen: (r) => {
      room = r;
      // Podpinamy obie odmiany zdarzenia - oficjalny przyklad node-haxball uzywa
      // onAfterRoomLink (nie onRoomLink) i w praktyce to wlasnie ono odpala
      // niezawodnie. Zostawiamy tez onRoomLink na wypadek roznic miedzy wersjami.
      room.onRoomLink = onRoomLink;
      room.onAfterRoomLink = onRoomLink;

      const stadiums = Utils.getDefaultStadiums();
      room.setCurrentStadium(stadiums.find(s => s.name === "Classic"));
      room.fakePlayerJoin(RED_ID, "Red", "pl", "🔴", "fake-conn-1", "fake-auth-1");
      room.fakePlayerJoin(BLUE_ID, "Blue", "pl", "🔵", "fake-conn-2", "fake-auth-2");
      room.setPlayerTeam(RED_ID, 1);
      room.setPlayerTeam(BLUE_ID, 2);
      room.setTimeLimit(0);
      room.setScoreLimit(0);
      room.startGame();

      room.onGameStop = () => {
        console.log("Mecz się zatrzymał - odpalam ponownie automatycznie.");
        room.startGame();
      };

      room.onTeamGoal = (teamId, goalId, goal) => {
        // Czerwoni atakują bramkę po x>0, niebiescy po x<0. Sprawdzamy realną
        // pozycję bramki (nie samo zdarzenie), żeby poprawnie przypisać samobóje
        // drużynie przeciwnej - dokładnie jak w normalnych zasadach.
        const positiveSide = goal && goal.p0 && goal.p0.x > 0;
        scoredBy = positiveSide ? 1 : 2;
      };

      room.onGameTick = () => {
        if (tickWaiters.length === 0) return;
        const state = readState();
        if (!state) return; // discs jeszcze nie gotowe - spróbujemy przy następnym ticku
        const waiters = tickWaiters;
        tickWaiters = [];
        waiters.forEach((resolve) => resolve(state));
      };

      onReady({
        reset() {
          scoredBy = null;
          // Losowe odchylenie pozycji startowej - idealna symetria (piłka dokładnie
          // na środku, boty dokładnie naprzeciw) sprawiała, że oba boty (ten sam
          // wyuczony model) podchodziły lustrzanie identycznie i wpadały w pat przy
          // piłce zamiast się rozstrzygnąć. Losowe przesunięcie łamie tę symetrię.
          const jitter = () => (Math.random() - 0.5) * 80; // +/-40 jednostek
          // Znikoma niezerowa prędkość piłki, żeby zdjąć barierę "kickoff" (silnik
          // czeka na ruch piłki, nie interesuje go kto go spowodował).
          room.setDiscProperties(0, { x: BALL_SPAWN.x + jitter(), y: BALL_SPAWN.y + jitter(), xspeed: 0.05, yspeed: 0.05 });
          room.setPlayerDiscProperties(RED_ID, { x: RED_SPAWN.x + jitter(), y: RED_SPAWN.y + jitter(), xspeed: 0, yspeed: 0 });
          room.setPlayerDiscProperties(BLUE_ID, { x: BLUE_SPAWN.x + jitter(), y: BLUE_SPAWN.y + jitter(), xspeed: 0, yspeed: 0 });
          room.fakeSendPlayerInput(0, RED_ID);
          room.fakeSendPlayerInput(0, BLUE_ID);
          if (room.gameState) {
            room.gameState.state = 1; // GamePlayState.Playing
          }
          return readState();
        },
        step(redDirX, redDirY, redKick, blueDirX, blueDirY, blueKick) {
          room.fakeSendPlayerInput(Utils.keyState(redDirX, redDirY, redKick), RED_ID);
          room.fakeSendPlayerInput(Utils.keyState(blueDirX, blueDirY, blueKick), BLUE_ID);
          return new Promise((resolve) => tickWaiters.push(resolve));
        }
      });
    }
  });
}

module.exports = { createEnv1v1 };
