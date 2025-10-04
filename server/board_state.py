from dataclasses import dataclass, field
from typing import List, Dict
from player_state import PlayerColor, Pawn
   
@dataclass
class Board:
    """Represents the game board, including all areas."""
    
    # Using a dictionary to hold pawns in each player's start area
    start_areas: Dict[PlayerColor, List[Pawn]] = field(default_factory=dict)

    safe_track: Dict[PlayerColor, List[Pawn | None]] = field(default_factory=dict)

    # The main track is a list of 52 squares
    main_track: List[Pawn | None] = field(default_factory=lambda: [None] * (53)) # 53 marks entry to safe area]
    
    # Color pattern for the main track (used for reward calculations)
    colors: List[str] = field(default_factory=lambda: ["Green", "Red", "Yellow", "Blue"]) 

    def __post_init__(self):
        """Initializes the board areas for each player color."""
        for color in PlayerColor:
            self.start_areas[color] = []
            self.safe_track[color] = [None] * 6