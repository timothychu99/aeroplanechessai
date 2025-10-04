import json
import uuid
import random
from datetime import datetime
from typing import List, Dict

from player_state import Player, PlayerColor
from board_state import Board
from action_state import roll_dice, get_available_pawns, move_pawn

def snapshot_state(players: List[Player], board: Board) -> Dict:
    """Create a snapshot dict describing pawns positions and occupied positions."""
    state = {}
    for i, player in enumerate(players, start=1):
        key = f"player_{i}_pawns"
        state[key] = [pawn.tile_position for pawn in player.pawns]

    occupied: Dict[str, List[str]] = {}
    # include all pawns (main track, safe track, finished)
    for player in players:
        for pawn in player.pawns:
            pos = pawn.tile_position
            if pos is None:
                continue
            # Only include positions that are > 0 (0 is start area)
            if pos > 0:
                occupied.setdefault(str(pos), []).append(pawn.id.replace(' ', '_'))

    state["occupied_positions"] = occupied
    return state


def collect_game_trace(num_games: int = 1, max_turns: int = 1000, out_file: str = "traces.jsonl"):
    traces = []
    for g in range(1, num_games + 1):
        game_id = f"G{g:03d}"
        # initialize players and board
        players = [Player(color=PlayerColor.RED), Player(color=PlayerColor.YELLOW), Player(color=PlayerColor.BLUE), Player(color=PlayerColor.GREEN)]
        board = Board()
        for p in players:
            board.start_areas[p.color] = p.pawns.copy()

        turn_number = 0
        current_idx = 0
        while turn_number < max_turns:
            turn_number += 1
            player = players[current_idx]

            # 1. roll (support combos by optionally using a list; here single roll)
            dice_roll = roll_dice()

            # snapshot before
            state_before = snapshot_state(players, board)

            # 2. get legal pawns
            available_pawns = get_available_pawns(player, dice_roll)
            legal_indices = [i + 1 for i, pawn in enumerate(player.pawns) if pawn in available_pawns]

            action_index = None
            subflags = {"capture": False, "finish": False, "hanger": False}
            reward = 0

            if available_pawns:
                # choose randomly
                chosen = random.choice(available_pawns)
                # record action as 1-based pawn index from player's pawn list
                action_index = player.pawns.index(chosen) + 1

                # record opponent positions before to detect capture
                opponent_positions_before = {op.id: op.tile_position for pl in players for op in pl.pawns if pl is not player}


                initial_start_area = len(board.start_areas[player.color])

                # perform move and get subflags
                flags = move_pawn(chosen, player, dice_roll, board)

                # merge flags into subflags
                if flags.get("capture"):
                    subflags["capture"] = True
                if flags.get("hopped_color_tile"):
                    subflags["hopped_color_tile"] = True
                if flags.get("hopped_occupied"):
                    subflags["hopped_occupied"] = True
                if flags.get("super_jump"):
                    subflags["super_jump"] = True

                # detect finish for moving pawn
                if chosen.landed:
                    subflags["finish"] = True

                # detect hanger: more than one pawn on same tile as chosen (friendly stack)
                if len(board.start_areas[player.color]) < initial_start_area:
                    subflags["hanger"] = True

                # base reward for making a move
                reward += 1
                # reward adjustments from flags
                if subflags.get("capture"):
                    reward += 60
                if subflags.get("hanger"):
                    reward += 30
                if subflags.get("finish"):
                    reward += 100
                if subflags.get("hopped_color_tile"):
                    reward += 30
                if subflags.get("hopped_occupied"):
                    reward += 30
                if subflags.get("super_jump"):
                    reward += 60
            else:
                # no legal moves
                action_index = -1
                reward = 0

            state_after = snapshot_state(players, board)

            record = {
                "game_id": game_id,
                "turn_number": turn_number,
                "player_turn": current_idx,
                "dice_roll": dice_roll,
                "legal_pawns": legal_indices,
                "action": action_index,
                "state_before": state_before,
                "state_after": state_after,
                "reward": reward,
                "subreward_flags": subflags,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

            traces.append(record)

            # check win
            if any(p.finished_pawns >= 4 for p in players):
                break

            if (dice_roll != 6):
                # next player
                current_idx = (current_idx + 1) % len(players)

        # write traces to file incrementally
        with open(out_file, "a", encoding="utf-8") as fh:
            for rec in traces:
                fh.write(json.dumps(rec) + "\n")
        print(f"Wrote {len(traces)} turns for {game_id} to {out_file}")


if __name__ == "__main__":
    collect_game_trace(num_games=150, max_turns=500, out_file="traces.jsonl")
