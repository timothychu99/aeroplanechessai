"""Play vs learned model: human vs robot.

Usage:
    python3 play_vs_model.py --model models/dqn.pth --human-color Red

Human will be prompted to choose a pawn when they have legal moves. Robot uses the DQN model to pick actions.
"""
import argparse
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


def print_board(players):
    print("P1:", [p.tile_position for p in players[0].pawns])
    print("P2:", [p.tile_position for p in players[1].pawns])


def human_choose(available, player):
    print("Your available pawns:")
    for i, pawn in enumerate(available):
        print(f"[{i}] {pawn.id} at pos {pawn.tile_position}")
    while True:
        try:
            choice = int(input("Choose pawn index (or -1 to pass): "))
            if choice == -1:
                return None
            if 0 <= choice < len(available):
                return available[choice]
        except Exception:
            pass
        print("Invalid choice")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/dqn.pth")
    # allow case-insensitive color names/values (e.g. red, Red, RED)
    parser.add_argument("--human-color", default="Red")
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if torch is None:
        print("PyTorch is required to run the robot. Install it first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qnet = load_model(args.model, device)

    # Map the provided human color (case-insensitive) to the PlayerColor enum.
    human_color_str = args.human_color
    try:
        human_color = next(
            c for c in PlayerColor
            if c.value.lower() == human_color_str.lower() or c.name.lower() == human_color_str.lower()
        )
    except StopIteration:
        valid = ", ".join([f"{c.value}" for c in PlayerColor])
        print(f"Invalid human color '{human_color_str}'. Valid values: {valid}")
        return

    # order: players[0] is RED, players[1] is BLUE (consistent with trainer)
    players = [Player(color=PlayerColor.RED), Player(color=PlayerColor.BLUE)]
    board = Board()
    for p in players:
        board.start_areas[p.color] = p.pawns.copy()

    # determine indices
    human_idx = 0 if players[0].color == human_color else 1
    robot_idx = 1 - human_idx

    current_idx = 0
    turn = 0
    while turn < args.max_turns:
        turn += 1
        player = players[current_idx]
        dice = roll_dice()

        if args.render:
            print(f"\nTurn {turn} - {player.color.value} rolled {dice}")

        available = get_available_pawns(player, dice)
        legal_indices = [i + 1 for i, pawn in enumerate(player.pawns) if pawn in available]

        chosen_pawn = None
        flags = None

        if current_idx == human_idx:
            # human's turn
            if not available:
                if args.render:
                    print("No legal moves. Passing.")
            else:
                chosen_pawn = human_choose(available, player)
                if chosen_pawn is None:
                    if args.render:
                        print("You chose to pass.")
                else:
                    flags = move_pawn(chosen_pawn, player, dice, board)
        else:
            # robot's turn
            if not available:
                if args.render:
                    print("Robot has no legal moves. Passing.")
            else:
                state_before = trainer.state_vector_from_snapshot({
                    "player_1_pawns": [pp.tile_position for pp in players[0].pawns],
                    "player_2_pawns": [pp.tile_position for pp in players[1].pawns],
                    "occupied_positions": {}
                }, current_idx)
                # pick action via qnet
                action = choose_action(qnet, state_before, legal_indices)
                if action is None or action == trainer.ACTION_SIZE - 1:
                    if args.render:
                        print("Robot chose to pass")
                else:
                    pawn_idx = action
                    chosen_pawn = player.pawns[pawn_idx]
                    if args.render:
                        print(f"Robot chose pawn {chosen_pawn.id}")
                    flags = move_pawn(chosen_pawn, player, dice, board)

        # render board and any flags
        if args.render:
            print_board(players)
            if flags:
                print("Flags:", flags)
            time.sleep(0.2)

        # check win
        if any(p.finished_pawns >= 4 for p in players):
            winner = next(p for p in players if p.finished_pawns >= 4)
            print(f"{winner.color.value} wins!")
            break

        # Next player unless rolled 6
        if dice != 6:
            current_idx = (current_idx + 1) % len(players)

    print("Game over")

if __name__ == "__main__":
    main()
