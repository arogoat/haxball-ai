const { Room, Utils } = require("node-haxball")();

const BOT_ID = 1;
const BALL_SPAWN = { x: 0, y: 0 };
const BOT_SPAWN = { x: -277.5, y: 0 };

function createEnv(token, onReady) {
  let room = null;
  let tickWaiters = [];
  let goalScored = false;
  let ownGoal = false;

  function readState() {
    const ball = room.getBall();
    const bot = room.getPlayerDisc(BOT_ID);
    const scored = goalScored;
    const own = ownGoal;
    goalScored = false;
    ownGoal = false;
    return {
      ballX: ball.pos.x, ballY: ball.pos.y,
      ballSpeedX: ball.speed.x, ballSpeedY: ball.speed.y,
      ballRadius: ball.radius,
      botX: bot.pos.x, botY: bot.pos.y,
      botSpeedX: bot.speed.x, botSpeedY: bot.speed.y,
      botRadius: bot.radius,
      goalScored: scored,
      ownGoal: own,
    };
  }

  Room.create({
    name: "haxball-ai-trening",
    noPlayer: true,
    maxPlayerCount: 10,
    token
  }, {
    storage: { player_name: "host" },
    onOpen: (r) => {
      room = r;

      room.onRoomLink = (link) => {
        console.log("Otwórz w przeglądarce, żeby oglądać:", link);
      };

      const stadiums = Utils.getDefaultStadiums();
      room.setCurrentStadium(stadiums.find(s => s.name === "Classic"));
      room.fakePlayerJoin(BOT_ID, "Bot", "pl", "🤖", "fake-conn", "fake-auth");
      room.setPlayerTeam(BOT_ID, 1);
      // Domyślnie mecz kończy się po 3 minutach LUB 3 golach - potem klatki przestają
      // lecieć i cały trening wisi w nieskończoność. Wyłączamy oba limity (0 = bez limitu).
      room.setTimeLimit(0);
      room.setScoreLimit(0);
      room.startGame();

      // Zabezpieczenie: gdyby mecz mimo wszystko się zatrzymał (np. z innego powodu),
      // odpalamy go ponownie automatycznie, zamiast wisieć bez końca.
      room.onGameStop = () => {
        console.log("Mecz się zatrzymał - odpalam ponownie automatycznie.");
        room.startGame();
      };

      room.onTeamGoal = (teamId, goalId, goal) => {
        // Bramka przeciwnika bota jest po stronie x>0 (bot spawnuje po lewej, x<0).
        // Sprawdzamy realną pozycję bramki, w której wylądowała piłka - nie ufamy
        // samemu zdarzeniu, bo odpala się też dla przypadkowego gola samobójczego.
        const scoredOnPositiveSide = goal && goal.p0 && goal.p0.x > 0;
        if (scoredOnPositiveSide) {
          goalScored = true;
        } else {
          ownGoal = true;
        }
      };

      room.onGameTick = () => {
        const waiters = tickWaiters;
        tickWaiters = [];
        waiters.forEach((resolve) => resolve(readState()));
      };

      onReady({
        reset() {
          goalScored = false;
          ownGoal = false;
          // Silnik zdejmuje barierę "kickoff" dopiero gdy prędkość piłki jest różna od
          // zera (kto by jej nie nadał) - zerowa prędkość (jak było wcześniej) utrzymuje
          // blokadę w nieskończoność, bo nie ma gracza w drużynie niebieskich, który by
          // "dotknął" piłkę. Dajemy więc piłce znikomą, ale niezerową prędkość startową.
          room.setDiscProperties(0, { x: BALL_SPAWN.x, y: BALL_SPAWN.y, xspeed: 0.05, yspeed: 0.05 });
          room.setPlayerDiscProperties(BOT_ID, { x: BOT_SPAWN.x, y: BOT_SPAWN.y, xspeed: 0, yspeed: 0 });
          room.fakeSendPlayerInput(0, BOT_ID);
          if (room.gameState) {
            room.gameState.state = 1; // GamePlayState.Playing - zostawiamy jako dodatkowe zabezpieczenie
          }
          return readState();
        },
        step(dirX, dirY, kick) {
          const keyState = Utils.keyState(dirX, dirY, kick);
          room.fakeSendPlayerInput(keyState, BOT_ID);
          return new Promise((resolve) => tickWaiters.push(resolve));
        }
      });
    }
  });
}

module.exports = { createEnv };