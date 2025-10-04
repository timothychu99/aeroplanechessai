from statemachine import StateMachine, State
from action_state import roll_dice, get_available_pawns, move_pawn
import time
import random

class ChessStateMachine(StateMachine):
    init = State(initial=True)
    diceroll = State()
    availablemoves = State()
    highlightmoves = State()
    selectpawn = State()
    checkwin = State()
    nextturn = State()

    start_game = init.to(diceroll)
    roll = diceroll.to(availablemoves)
    show_moves = availablemoves.to(highlightmoves, cond="pawns_available") | availablemoves.to(nextturn, cond="no_pawns_available")
    wait_for_pawn = highlightmoves.to(selectpawn, cond="robot_player") | highlightmoves.to(selectpawn, cond="pawn_selected")
    move = selectpawn.to(checkwin)
    cycle = checkwin.to(nextturn)
    again = nextturn.to(diceroll)

    def __init__(self, game, move_func=None, gid=None):
        self.game = game
        self.available_pawns = []
        self.gid = None
        self._pawn_id = None
        self.move_func = move_func
        self.show_states = False
        self.curr_move_player_idx = -1
        self.winner = False
        super().__init__()

    # --- Conditions ---
    def pawn_selected(self):
        ids = [pawn.id for pawn in self.available_pawns]
        return self._pawn_id in ids
    def pawns_available(self):
        return len(self.available_pawns) > 0
    def no_pawns_available(self):
        return len(self.available_pawns) == 0
    def robot_player(self):
        return self.game.current_idx % 4 != 0  # Assuming 1 is robot

    # --- Hooks ---
    def on_enter_checkwin(self):
        if self.game.players[self.curr_move_player_idx].finished_pawns == 4:
            print(f"🏆 Player {self.curr_move_player_idx} wins!")
            self.winner = True  # Force to final state
        else:
            print("No winner yet, continuing...")
           
    def on_enter_diceroll(self):
        self.curr_move_player_idx = self.game.current_idx
        self.game.dice = random.randint(1, 6)
        print(f"🎲 Rolled dice: {self.game.dice} for player {self.game.current_idx}")
        
    def on_enter_availablemoves(self):
        self.available_pawns = get_available_pawns(self.game.players[self.game.current_idx], self.game.dice)
        print("Available pawns:", self.available_pawns)

        self.show_states = True
        time.sleep(2)  # Pause before next turn
        self.show_states = False

        if not self.available_pawns:
            print("No available moves, skipping turn.")
            self.game.current_idx = (self.game.current_idx + 1) % len(self.game.players)
        else:
            print("Moves available, highlighting...")

    def on_enter_highlightmoves(self):
        print("Highlighting moves on board...")
        # Delay before allowing selection
        if self.game.current_idx % 4 != 0:  # Robot player
            print("Robot player, auto-selecting pawn...")
            self.wait_for_pawn()
        time.sleep(1.5)  # <-- artificial pause for animation
        print("Now waiting for pawn selection...")

    def on_enter_selectpawn(self):
#        print(f"Selected pawn {self._pawn_id}, moving it...")
        self.move_func(self.game, self.available_pawns, self.gid)
#        api_play_step(self.game, [self._pawn_id])

    def on_enter_nextturn(self):
        if self.game.dice == 6:
            print("Rolled a 6, player gets another turn!")
        print(f"➡️ Next turn: Player {self.game.current_idx}")
        self.available_pawns = []
        self._pawn_id = None

    # --- External trigger for human input ---
    def select_pawn(self, pawn_id):
        self._pawn_id = pawn_id
        if self.current_state == self.highlightmoves and self.pawn_selected():
            # Force transition to selectpawn when pawn chosen
            self.wait_for_pawn()

## --- Example usage ---
#sm = ChessStateMachine(game)
#sm.start_game()   # init -> diceroll
#sm.roll()         # diceroll -> availablemoves
#sm.show_moves()   # availablemoves -> highlightmoves (highlights + pause)
## ... UI shows moves, waits ...
#sm.select_pawn("pawn_3")  # external input from client
#sm.move()         # selectpawn -> movingpawn
