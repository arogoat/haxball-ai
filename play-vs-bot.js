// GRAJ Z BOTEM (1v1 classic). Hostuje pokoj haxball, w ktorym bot (nasza
// wytrenowana siec) gra czerwonymi, a Ty dolaczasz w przegladarce i grasz
// niebieskimi. Bot jest sterowany przez bot_server.py (Python+SB3) przez
// lokalne TCP - odpal go NAJPIERW:
//     ./venv/bin/python bot_server.py
// potem w drugim terminalu:
//     node play-vs-bot.js
// Link do pokoju pojawi sie w konsoli (i w room-links-vs-bot.txt).
//
// BONUS: mecz jest nagrywany do recordings-vs-bot/*.hbr2 - kazda Twoja gra
// laduje jako material do imitation learning (ten sam pipeline co nagrania z
// MrREPLAY: tools/convert-replay.js).
const fs = require("fs");
const net = require("net");
const { Room, Utils } = require("node-haxball")();
const { TOKENS_1V1 } = require("./tokens.js");

const TOKEN = process.argv[2] || TOKENS_1V1[0];
const BOT_PORT = 5555;
const ACTION_REPEAT = 5;
const RED = 1, BLUE = 2;
// wysokie ID - prawdziwi gracze dostaja ID sekwencyjnie od 1, wiec 99 nie
// koliduje (przy niskim ID czlowiek "wchodzil w bota" i drozyny sie mieszaly)
const BOT_ID = 99;

// --- polaczenie z serwerem bota (Python) ---
let botAction = [1, 1, 0]; // domyslnie: stoj, nie kopie (indeksy {0,1,2},{0,1})
let prevAction = [1, 1, 0];
let waiting = false;
const botConn = net.connect(BOT_PORT, "127.0.0.1", () => {
  console.log("Polaczono z serwerem bota (Python).");
});
let rbuf = "";
botConn.on("data", (d) => {
  rbuf += d.toString();
  let i;
  while ((i = rbuf.indexOf("\n")) >= 0) {
    const line = rbuf.slice(0, i); rbuf = rbuf.slice(i + 1);
    if (!line.trim()) continue;
    try { botAction = JSON.parse(line); waiting = false; } catch (e) { /* ignore */ }
  }
});
botConn.on("error", (e) => {
  console.error("BLAD polaczenia z botem:", e.message, "- czy bot_server.py dziala?");
});

function askBot(state) {
  if (waiting || botConn.destroyed) return;
  waiting = true;
  botConn.write(JSON.stringify(state) + "\n");
}

Room.create({ name: "Zagraj z AI (1v1)", noPlayer: true, maxPlayerCount: 6, token: TOKEN, showInRoomList: false }, {
  storage: { player_name: "host" },
  onRoomLink: (link) => {
    console.log("\n=== WEJDZ TU, ZEBY ZAGRAC Z BOTEM: ===\n" + link + "\n");
    try { fs.appendFileSync("room-links-vs-bot.txt", new Date().toISOString() + " " + link + "\n"); } catch (e) {}
  },
  onOpen: (room) => {
    // Pewne pobranie linka: zdarzenie onRoomLink bywa zawodne, a room.link to
    // zwykla wlasciwosc. Odpytujemy co 2s az bedzie kompletny (?c=... nie null).
    let linkShown = false, attempts = 0;
    const linkPoll = setInterval(() => {
      if (linkShown) { clearInterval(linkPoll); return; }
      if (room.link && !room.link.includes("null")) {
        linkShown = true; clearInterval(linkPoll);
        console.log("\n=== WEJDZ TU, ZEBY ZAGRAC Z BOTEM: ===\n" + room.link + "\n");
        try { fs.appendFileSync("room-links-vs-bot.txt", new Date().toISOString() + " " + room.link + "\n"); } catch (e) {}
        return;
      }
      if (++attempts === 30) {
        console.log("UWAGA: pokoj nie dostal linka po 60s - token najpewniej wygasl. Wygeneruj swiezy: https://www.haxball.com/headlesstoken");
      }
    }, 2000);

    const stadiums = Utils.getDefaultStadiums();
    room.setCurrentStadium(stadiums.find((s) => s.name === "Classic"));
    room.setTimeLimit(0);
    room.setScoreLimit(0);

    // bot jako czerwony
    room.fakePlayerJoin(BOT_ID, "🤖 Bot", "pl", "🔴", "bot-conn", "bot-auth");
    room.setPlayerTeam(BOT_ID, RED);

    // Gra startuje OD RAZU (bot vs pusta niebieska) i leci caly czas - dzieki
    // temu wchodzacy czlowiek od razu gra, zamiast siedziec w niewystartowanym
    // meczu (co wygladalo jak spectator).
    try { room.startRecording(); } catch (e) {}
    room.startGame();
    // gdyby gra sie zatrzymala (np. reset), odpal ja ponownie
    room.onGameStop = () => { setTimeout(() => { try { room.startGame(); } catch (e) {} }, 500); };

    // czlowiek -> niebiescy zaraz po dolaczeniu
    room.onPlayerJoin = (p) => {
      if (p.id !== BOT_ID) { try { room.setPlayerTeam(p.id, BLUE); } catch (e) {} }
    };

    // zamiatacz samo-korygujacy: bot ZAWSZE czerwony, kazdy inny ZAWSZE niebieski
    setInterval(() => {
      room.players.forEach((p) => {
        const want = p.id === BOT_ID ? RED : BLUE;
        if (!p.team || p.team.id !== want) {
          try { room.setPlayerTeam(p.id, want); } catch (e) {}
        }
      });
    }, 1000);
    room.onPlayerLeave = () => {
      const humans = room.players.filter((p) => p.id !== BOT_ID && p.team && p.team.id !== 0);
      if (humans.length === 0) {
        try { const rec = room.stopRecording(); if (rec) saveRecording(rec); } catch (e) {}
      }
    };
    let tick = 0;
    room.onGameTick = () => {
      const ball = room.getBall();
      const bot = room.getPlayerDisc(BOT_ID);
      // przeciwnik = pierwszy zywy czlowiek (niebieski)
      const humanP = room.players.find((p) => p.id !== BOT_ID && p.team && p.team.id === BLUE);
      const opp = humanP ? room.getPlayerDisc(humanP.id) : null;
      if (!ball || !bot) return;

      if (tick % ACTION_REPEAT === 0) {
        askBot({
          team: RED,
          ballX: ball.pos.x, ballY: ball.pos.y, ballVX: ball.speed.x, ballVY: ball.speed.y,
          selfX: bot.pos.x, selfY: bot.pos.y, selfVX: bot.speed.x, selfVY: bot.speed.y,
          oppX: opp ? opp.pos.x : 0, oppY: opp ? opp.pos.y : 0,
          oppVX: opp ? opp.speed.x : 0, oppVY: opp ? opp.speed.y : 0,
          prev: prevAction,
        });
        prevAction = botAction;
      }
      const dirX = botAction[0] - 1;
      const dirY = botAction[1] - 1;
      const kick = botAction[2] === 1;
      room.fakeSendPlayerInput(Utils.keyState(dirX, dirY, kick), BOT_ID);
      tick++;
    };
  },
});

function saveRecording(uint8) {
  try {
    fs.mkdirSync("recordings-vs-bot", { recursive: true });
    const name = `recordings-vs-bot/vsbot-${Date.now()}.hbr2`;
    fs.writeFileSync(name, Buffer.from(uint8));
    console.log("Zapisano nagranie:", name);
  } catch (e) { console.error("Nie udalo sie zapisac nagrania:", e.message); }
}
