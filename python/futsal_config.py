# Stale specyficzne dla mapy Futsal x7 (wyciagniete z prawdziwej geometrii mapy,
# tools/extract-stadium.js). Classic mial bramke na x=370 i boisko ~420x200;
# futsal jest ~3x wiekszy, wiec skale normalizacji i pozycje bramek sa inne.
import os

STADIUM_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stadiums", "Futsal-x7-by-Bazinga-from-HaxMaps.json5")

# Geometria (z definicji mapy)
FIELD_HALF_X = 1250.0     # zakres boiska w x: +-1250
FIELD_HALF_Y = 660.0      # zakres boiska w y: +-660
GOAL_X = 1207.0           # linia bramkowa
GOAL_HALF_HEIGHT = 95.0   # polowa wysokosci bramki (slupki na y=+-95)

# Skale normalizacji obserwacji (~polowa boiska / typowa predkosc)
POS_SCALE = 1250.0
VEL_SCALE = 8.0           # futsalowa pilka/gracze szybsze niz na classic

# Spawny do losowego startu (wewnatrz boiska, z marginesem od band)
SPAWN_HALF_X = 1150.0
SPAWN_HALF_Y = 560.0
BALL_SPAWN_HALF_X = 1050.0
BALL_SPAWN_HALF_Y = 480.0
