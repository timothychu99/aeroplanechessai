import time

try:
    import torch
except Exception:
    torch = None

from player_state import Player, PlayerColor
from board_state import Board
from action_state import roll_dice, get_available_pawns, move_pawn

# reuse utilities from play_with_model
from play_with_model import load_model, choose_action, trainer

# ------------------------
# Hardcoded config
# ------------------------
MODEL_PATH = "models/4p_epoch30.pth"
HUMAN_COLOR = "Red"   # could be "Blue"
MAX_TURNS = 500
RENDER = False

# Load robot model once
if torch is None:
    print("⚠️ PyTorch is required to run the robot. Install it first.")
    qnet = None
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qnet = load_model(MODEL_PATH, device)


def print_board(players):
    print("P1:", [p.tile_position for p in players[0].pawns])
    print("P2:", [p.tile_position for p in players[1].pawns])


#def human_choose(available):
#    """Prompt human to pick a pawn (console fallback)."""
#    print("Your available pawns:")
    

def robot_choose(players, current_idx, available):
    """Choose an action using the trained DQN model."""
    if not available or qnet is None:
        return None

    state_before = trainer.state_vector_from_snapshot({
        "player_1_pawns": [pp.tile_position for pp in players[0].pawns],
        "player_2_pawns": [pp.tile_position for pp in players[1].pawns],
        "occupied_positions": {}
    }, current_idx)

    legal_indices = [i + 1 for i, pawn in enumerate(players[current_idx].pawns) if pawn in available]

    action = choose_action(qnet, state_before, legal_indices)

    if action is None or action == trainer.ACTION_SIZE - 1:
        return -1

    return action
