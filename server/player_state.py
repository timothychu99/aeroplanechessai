from dataclasses import dataclass, field
from enum import Enum
from typing import List

"""Module for managing player states in a board game."""

class PlayerColor(Enum):
    """Enumeration for player colors."""
    RED = "Red"
    GREEN = "Green"
    BLUE = "Blue"
    YELLOW = "Yellow"
        
@dataclass
class Pawn:
    """Data class representing a pawn in the game."""
    id: str
    tile_position: int = -2  # Starting position on the board
    in_start_area: bool = True
    in_main_track: bool = False
    in_safe_area: bool = False
    landed: bool = False  # Indicates if the pawn has reached the final position

@dataclass
class Player:
    """Data class representing a player in the game."""
    color: PlayerColor

    pawns: List[Pawn] = field(init=False)

    safe_track_finish_index: int = 5

    finished_pawns: int = 0

    PAWNS_TOTAL = 4

    def __post_init__(self):
        self.pawns = [Pawn(f"{self.color.value}_{pawn_id}") for pawn_id in range(1, self.PAWNS_TOTAL + 1)]

    def get_pawn_ids(self):
        return [pawn.id for pawn in self.pawns]
