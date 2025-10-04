import random
from dataclasses import dataclass, field
from board_state import Board
from player_state import Player, Pawn
from typing import List
from player_state import PlayerColor, Pawn

def roll_dice():
    """Simulates rolling a six-sided dice and returns the total."""
    return random.randint(1, 6)

# for getting correct index of board
def get_board_idx(player_color: PlayerColor, tile_position : int) -> int:
    if tile_position == -1:
        return None
    if player_color == "Red":
        return tile_position 
    if player_color == "Yellow":
        return (tile_position + 13) % 52        
    if player_color == "Blue":
        return (tile_position + 26) % 52 
    # means it is green
    return (tile_position + 39) % 52 

def get_main_tile_color(player_color: PlayerColor):
    if player_color == "Green":
        return 0
    if player_color == "Red":
        return 1
    if player_color == "Yellow":
        return 2
    if player_color == "Blue":
        return 3

def jump_tile_or_eat(pawn: Pawn, board: Board, next_tile_position: int):
    # prohibits double jumping color tile
    jumped_color_tile = False
    flags = {"capture": False, "hopped_color_tile": False, "hopped_occupied": False, "super_jump": False}

    # checks for jump tile, hopping occupant pawn, and eating
    while True:

        idx = get_board_idx(pawn.id.split('_')[0], next_tile_position)
        # guard against invalid idx
        if idx < 0 or idx >= len(board.main_track):
            return next_tile_position, flags

        occupant = board.main_track[idx]
        color_tile = (idx % 4 == get_main_tile_color(pawn.id.split('_')[0]))

        if next_tile_position < 50 and occupant is not None and occupant.id.split('_')[0] != pawn.id.split('_')[0]:
            # Eat enemy pawn
            demote_pawn_to_start_area(occupant, board)
            flags["capture"] = True
            continue

        elif (next_tile_position == 17):
            print(f"Pawn {pawn.id} landed on a SUPER jump tile! Jumping to 29")
            flags["super_jump"] = True
            next_tile_position += 12
            jumped_color_tile = True
            continue

        elif occupant is not None and occupant.id.split('_')[0] == pawn.id.split('_')[0]:
            # Jump over own pawn ONCE (don’t loop forever)
            next_tile_position += 4
            flags["hopped_occupied"] = True
            jumped_color_tile = True
            # continue loop to check next landing spot
            continue

        elif color_tile and not jumped_color_tile:
            # Jump forward 4
            next_tile_position += 4
            flags["hopped_color_tile"] = True
            jumped_color_tile = True
            # continue loop to allow double jump
            continue

        else:
            # Normal stop
            return next_tile_position, flags

def promote_pawn_to_main_track(pawn: Pawn, board: Board) -> None:
    """Promotes a pawn from the start area to the main track."""
    pawn.in_start_area = False
    pawn.in_main_track = True

    pawn.tile_position = -1  # position 1 is the entry point to the main track
    board.start_areas[PlayerColor(pawn.id.split('_')[0])].remove(pawn)

def demote_pawn_to_start_area(pawn: Pawn, board: Board) -> None:
    """Demotes a pawn back to the start area."""
    pawn.in_start_area = True
    pawn.in_main_track = False
    pawn.in_safe_area = False
    board.main_track[get_board_idx(pawn.id.split('_')[0], pawn.tile_position)] = None

    pawn.tile_position = -2  # Reset position to start area
    board.start_areas[PlayerColor(pawn.id.split('_')[0])].append(pawn)

def promote_pawn_to_safe_area(pawn: Pawn, board: Board) -> None:
    """Promotes a pawn to the safe area."""
    pawn.in_main_track = False
    pawn.in_safe_area = True

    board.main_track[get_board_idx(pawn.id.split('_')[0], pawn.tile_position)] = None

def promote_pawn_to_finished(pawn: Pawn) -> None:
    """Promotes a pawn to the finished state."""
    pawn.in_safe_area = False
    pawn.landed = True

def get_available_pawns(player: Player, steps: int) -> List[Pawn]:
    """Returns a list of pawns that are not in the home area."""

    # Pawns in start area can only move out if a 12 is rolled
    if steps == 6:
        return [pawn for pawn in player.pawns if not pawn.landed]
    
    return [pawn for pawn in player.pawns if not pawn.in_start_area and not pawn.landed]

def move_pawn(pawn : Pawn, player : Player, steps: int, board : Board) -> None:
    """Moves a pawn a specified number of steps on the board."""

    # start area promotion
    if pawn.in_start_area and steps == 6:
        promote_pawn_to_main_track(pawn, board)
        #board.main_track[get_board_idx(pawn.id.split('_')[0], pawn.tile_position)] = pawn
        print(f"Pawn {pawn.id} promoted from start area. Roll Dice Again!")
        return {"capture": False, "hopped_color_tile": False, "hopped_occupied": False}
    
    # main track movement
    next_tile_position = pawn.tile_position + steps

    if pawn.in_main_track and next_tile_position < 50:
        pre_idx = get_board_idx(pawn.id.split('_')[0], pawn.tile_position)
        if pre_idx is not None:
            board.main_track[get_board_idx(pawn.id.split('_')[0], pawn.tile_position)] = None

        next_tile_position, flags = jump_tile_or_eat(pawn, board, next_tile_position)

        if (next_tile_position < 50):
            board.main_track[get_board_idx(pawn.id.split('_')[0], next_tile_position)] = pawn
            pawn.tile_position = next_tile_position


    # Safe area movement and if jumping makes it end up here
    if next_tile_position >= 50:

        # check if promotion needed and remove previous pawn position
        if pawn.in_main_track:
            promote_pawn_to_safe_area(pawn, board)
        else:
            board.safe_track[player.color][pawn.tile_position - 50] = None
        
        if next_tile_position == 55:
            promote_pawn_to_finished(pawn)
            print(f"Pawn {pawn.id} finished")
            player.finished_pawns += 1

        period = 55   # = 55 = end tile-position, as it is 0 indexed
        pos = (next_tile_position) % period

        if next_tile_position == 55:
            final_idx = 55
        elif next_tile_position < 55:
            final_idx = pos
        else:
            final_idx = 55 - (pos)

        board.safe_track[player.color][final_idx - 50] = pawn
        pawn.tile_position = final_idx

    # default return flags if not previously set
    try:
        return flags
    except NameError:
        return {"capture": False, "hopped_color_tile": False, "hopped_occupied": False, "super_jump": False}


@dataclass
class ActionState:
    """Represents the state of an action in the game."""
    roll: int = field(default_factory=roll_dice)
    player: Player = None
    move_pawn_id: str = ""
    action_description: str = ""
