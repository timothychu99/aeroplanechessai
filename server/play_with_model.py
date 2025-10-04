"""Play games using a trained DQN model.

Usage:
    python3 play_with_model.py --model models/dqn.pth --games 5 --max-turns 500

If no model exists the script will exit with instructions to train one (see ai_train.py).
"""
import argparse
import random
import os
import time

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None

from player_state import Player, PlayerColor
from board_state import Board
from action_state import roll_dice, get_available_pawns, move_pawn

# reuse state encoding from ai_train
import ai_train as trainer


class QNet(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


def load_model(path: str, device):
    # Try exact path first
    candidate = path
    if not os.path.exists(candidate):
        # try common alternate extensions
        base, ext = os.path.splitext(path)
        for alt in (".pth", ".pt"):
            alt_path = base + alt
            if os.path.exists(alt_path):
                candidate = alt_path
                break
    # fallback: pick newest model in models/ directory
    if not os.path.exists(candidate):
        models_dir = os.path.dirname(path) or "models"
        if os.path.isdir(models_dir):
            files = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.endswith((".pth", ".pt"))]
            if files:
                files.sort(key=os.path.getmtime, reverse=True)
                candidate = files[0]

    if not os.path.exists(candidate):
        raise FileNotFoundError(f"Model file not found: {path} (checked {candidate})")

    data = torch.load(candidate, map_location=device)
    # data may be the state_dict itself or a dict containing model_state and input_dim
    if isinstance(data, dict):
        state = data.get("model_state") or data.get("state_dict") or data
        input_dim = data.get("input_dim")
    else:
        state = data
        input_dim = None

    # If input_dim missing, try to infer from state tensor shapes
    action_dim = trainer.ACTION_SIZE
    if input_dim is None:
        # try to infer input dim from first Linear weight in state dict
        if isinstance(state, dict):
            for k, v in state.items():
                if "weight" in k and isinstance(v, torch.Tensor):
                    # weight shape: (out, in)
                    input_dim = v.shape[1]
                    break
    if input_dim is None:
        raise RuntimeError("Saved model missing input_dim; retrain with ai_train that saves input_dim or provide a compatible model")

    qnet = QNet(input_dim, action_dim).to(device)
    if isinstance(state, dict):
        qnet.load_state_dict(state)
    else:
        # assume it's a full state_dict
        qnet.load_state_dict(state)
    qnet.eval()
    print(f"Loaded model from {candidate}")
    return qnet


def choose_action(qnet, state_vec, available_pawns_indices, debug: bool = False):
    # state_vec : list/ndarray
    # create tensor on the same device as the model
    device = next(qnet.parameters()).device
    x = torch.tensor([state_vec], dtype=torch.float32, device=device)
    with torch.no_grad():
        q = qnet(x).squeeze(0).cpu().numpy()

    # mask illegal pawn actions
    legal = set(available_pawns_indices)
    # actions mapping: 0..3 -> pawn 1..4 indices, last -> pass
    best = None
    best_val = None

    # If there are available pawn moves, do NOT allow the pass action.
    pass_allowed = len(legal) == 0

    for a in range(trainer.ACTION_SIZE):
        if a == trainer.ACTION_SIZE - 1:  # pass action
            is_legal = pass_allowed
        else:
            is_legal = (a + 1) in legal
        if not is_legal:
            continue
        val = float(q[a])
        if best is None or val > best_val:
            best = a
            best_val = val
    return best


def run_game(qnet, max_turns=500, render=False, delay=0.2, debug=False):
    players = [Player(color=PlayerColor.RED), Player(color=PlayerColor.BLUE)]
    board = Board()
    for p in players:
        board.start_areas[p.color] = p.pawns.copy()

    current_idx = 0
    turn = 0
    history = []
    while turn < max_turns:
        turn += 1
        player = players[current_idx]
        dice = roll_dice()
        state_before = trainer.state_vector_from_snapshot({
            "player_1_pawns": [pp.tile_position for pp in players[0].pawns],
            "player_2_pawns": [pp.tile_position for pp in players[1].pawns],
            "occupied_positions": {}
        }, current_idx)

        available = get_available_pawns(player, dice)
        legal_indices = [i + 1 for i, pawn in enumerate(player.pawns) if pawn in available]

        action = None
        if available:
            action = choose_action(qnet, state_before, legal_indices, debug=debug)
            if action is None:
                action = trainer.ACTION_SIZE - 1
        else:
            action = trainer.ACTION_SIZE - 1

        # map action to pawn or pass
        if action == trainer.ACTION_SIZE - 1:
            chosen_pawn = None
            if render:
                print(f"Turn {turn}: {player.color.value} rolled {dice} -> pass")
        else:
            pawn_idx = action  # 0-based
            chosen_pawn = player.pawns[pawn_idx]
            if render:
                print(f"Turn {turn}: {player.color.value} rolled {dice} -> move pawn {chosen_pawn.id}")
            move_pawn(chosen_pawn, player, dice, board)

        state_after = trainer.state_vector_from_snapshot({
            "player_1_pawns": [pp.tile_position for pp in players[0].pawns],
            "player_2_pawns": [pp.tile_position for pp in players[1].pawns],
            "occupied_positions": {}
        }, current_idx)

        history.append({
            "turn": turn,
            "player": current_idx,
            "dice": dice,
            "action": action,
            "state_before": state_before,
            "state_after": state_after
        })

        if render:
            print("P1:", [p.tile_position for p in players[0].pawns])
            print("P2:", [p.tile_position for p in players[1].pawns])
            time.sleep(delay)

        if any(p.finished_pawns >= 4 for p in players):
            print("GAME WINNER {player}")
            break

        if dice != 6:
            current_idx = (current_idx + 1) % len(players)

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/dqn.pth")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if torch is None:
        print("PyTorch is required to run the player. Install it first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qnet = load_model(args.model, device)

    for g in range(args.games):
        hist = run_game(qnet, max_turns=args.max_turns, render=args.render, debug=args.debug)
        print(f"Game {g+1}: {len(hist)} turns")


if __name__ == "__main__":
    main()
