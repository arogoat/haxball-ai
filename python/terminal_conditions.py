# Poprawiony warunek konca epizodu po golu.
# Oryginalny GoalScoredCondition z haxballgym ma literowke:
#     self.red_score = current_state.blue_score   # powinno byc red_score!
#     self.blue_score = current_state.blue_score
# Skutek: po KAZDYM meczu z golem zapamietany wynik jest bledny i nastepny
# epizod konczy sie natychmiast (1-krokowe "mecze-widmo" widoczne w eval,
# a wczesniej po cichu zasmiecajace tez dane treningowe).
# Nasza wersja dodatkowo zapamietuje wynik przy reset() zamiast zakladac 0:0 -
# silnik NIE zeruje wyniku miedzy epizodami.
from haxballgym.utils.gamestates import GameState
from haxballgym.utils.terminal_conditions import TerminalCondition


class FixedGoalScoredCondition(TerminalCondition):
    def __init__(self):
        super().__init__()
        self.red_score = 0
        self.blue_score = 0

    def reset(self, initial_state: GameState):
        self.red_score = initial_state.red_score
        self.blue_score = initial_state.blue_score

    def is_terminal(self, current_state: GameState) -> bool:
        if (
            current_state.red_score != self.red_score
            or current_state.blue_score != self.blue_score
        ):
            self.red_score = current_state.red_score
            self.blue_score = current_state.blue_score
            return True
        return False
