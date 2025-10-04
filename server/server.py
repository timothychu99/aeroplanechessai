# server/server/py

#TODO: Fix pawns disappeaaring when moving out of start area

# Use flask for python server connection
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import json
import uuid
import torch
import ai_train as trainer

# game imports
from player_state import Player, PlayerColor
from board_state import Board
from action_state import roll_dice, get_available_pawns, move_pawn
from server_play_vs_model import robot_choose
import random
import time
from fsm import ChessStateMachine
import threading
from collections import deque
import queue

winner = -100
# for logs on client
log_buffer = deque(maxlen=10)

def add_log(message):
    print(message, flush=True)
    log_buffer.append(message)


GAME_FSMS = {}  # key: gid, value: ChessStateMachine
g_gid = None
pawn_id = None 
move_counter = 0

def get_pawn_id():
    global pawn_id
    return pawn_id

def set_pawn_id(pid):
    global pawn_id
    pawn_id = pid

def run_sm(sm):
    while not sm.winner:
        state = sm.current_state
        with app.app_context():
            if state == sm.diceroll:
                sm.roll()
            elif state == sm.availablemoves:
                sm.show_moves()
            elif state == sm.selectpawn:
                sm.move()
            elif state == sm.checkwin:
                sm.cycle()
            elif state == sm.nextturn:
                sm.again()
                
        time.sleep(.3)
    global winner
    winner = sm.game.current_idx - 1 % 4
    log = "Game Over! Player " + str((sm.game.current_idx - 1) % 4) + " " + str(sm.game.players[(sm.game.current_idx - 1) % 4].color.name) + " wins!"
    add_log(log)


# Game class starts board and snapshots player/pawn states
class Game:
    def __init__(self):
        # two-player game by default (Red vs Blue)
        self.players = [Player(color=PlayerColor.RED), Player(color=PlayerColor.YELLOW), Player(color=PlayerColor.BLUE), Player(color=PlayerColor.GREEN)]
        self.board = Board()
        # all pawns in player's start_areas
        for p in self.players:
            self.board.start_areas[p.color] = p.pawns.copy()
        self.current_idx = 0
        self.available_pawns = []
        self.dice = None

    def snapshot(self):
        # return list of players (color, pawns) and their pawn positions (pawn idx and board idx)
        players_list = []
        for p in self.players:
            pawns_list = []
            for pawn in p.pawns:
                # try to find pawn index in main_track
                try:
                    bidx = self.board.main_track.index(pawn)
                except ValueError:
                    bidx = None
                pawns_list.append({'id': pawn.id, 'tile_position': pawn.tile_position, 'board_idx': bidx})
            players_list.append({'color': p.color.value, 'pawns': pawns_list, 'finished_pawns': p.finished_pawns})
        return {'players': players_list, 'current_player': self.current_idx}

# Makes Game sessions to control multiple Games
class GameManager:
    def __init__(self):
        self.games = {}

    def create_game(self):
        gid = str(uuid.uuid4())
        g = Game()
        self.games[gid] = g
        return gid, g

    def get(self, gid):
        return self.games.get(gid)

#--------------Start the Game---------------
gm = GameManager()
sm = None
# Create the Flask app instance
app = Flask(__name__)

# This is the crucial part for connecting the two servers.
# It allows requests from our Next.js frontend (running on http://localhost:3000)
# to be accepted by our Flask backend (running on http://localhost:8080).
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

message_queue = queue.Queue()



@app.route('/api/play/stream')
def stream():
    def event_stream():
        while True:
            msg = message_queue.get()  # blocks until a message is available
            # ensure message is a JSON string
            if not isinstance(msg, str):
                try:
                    msg = json.dumps(msg)
                except Exception:
                    msg = str(msg)
            yield f"data: {msg}\n\n"
    return Response(event_stream(), mimetype='text/event-stream')


#TODO: Remove api/data api endpoint
@app.route('/api/play/data', methods=['GET'])
def get_play_data():
    # This data will be sent to the frontend
    global g_gid
    sm = GAME_FSMS.get(g_gid)

    if sm is None:
        return jsonify({'error': 'Game not started'}), 400
    
    data = {
        "dice": sm.game.dice,
        "available": [p.id for p in sm.available_pawns],
        "current_player": sm.game.players[sm.game.current_idx].color.value,
    }
    return jsonify(data)

@app.route('/api/log/latest', methods=['GET'])
def get_latest_logs():
    return jsonify({ "latest_log": log_buffer[-1] if log_buffer else ""})

# A simple API endpoint
@app.route('/api/data', methods=['GET'])
def get_data():
    # This data will be sent to the frontend

    global g_gid
    sm = GAME_FSMS.get(g_gid)

    if sm is None:
        return jsonify({'error': 'Game not started'}), 400
    
    data = {
        "message": sm.show_states
    }
    return jsonify(data)

@app.route('/api/play/start', methods=['POST'])
def api_play_start():
    """Start a new game and return the initial pawn positions."""
    global game
    global g_gid
    gid, game = gm.create_game()
    sm = ChessStateMachine(game=game, move_func=api_play_step, gid=gid)
    g_gid = gid
    sm.start_game()

    threading.Thread(target=run_sm, args=(sm,)).start()
    GAME_FSMS[g_gid] = sm

    # DEBUGGING
    return jsonify({'game_id': gid, 'state': game.snapshot(), 'current_player': game.players[game.current_idx].color.value, 'dice': game.dice})

@app.route('/api/play/choose', methods=['POST'])
def api_play_choose():

    data = request.get_json(force=True)
    pawn_id = data.get('pawn_id')
    log = "Received pawn_id: " + str(pawn_id)
    add_log(log)

    set_pawn_id(pawn_id)

    global g_gid
    sm = GAME_FSMS.get(g_gid)

    if sm is None:
        return jsonify({'error': 'Game not started'}), 400
    if sm.current_state == sm.highlightmoves:
        add_log("In highlightmoves state, proceeding...")
    else:
        log = "Not in selectpawn state, current state: " + str(sm.current_state)
        add_log(log)
        return jsonify({'error': 'Not in selectpawn state'}), 400

    threading.Thread(target=sm.select_pawn(pawn_id), args=(pawn_id,)).start()
    pawn_id = None  # reset after use
    GAME_FSMS[g_gid] = sm

    return jsonify({})

move_idx = {'player': None, 'chosen': None, 'before_board_idx': None, 'after_board_idx': None, 'dice': None}
@app.route('/api/play/step', methods=['POST'])
def api_play_step(game=None, available=None, gid=None):
    if not game:
        return jsonify({'error': 'game not found'}), 404

    pidx = game.current_idx
    player = game.players[pidx]
 
    last_move = {'player': pidx, 'player_color': player.color.value, 'dice': game.dice, 'moved': None, 'before_board_idx': None, 'after_board_idx': None, 'before_tile_position': None, 'after_tile_position': None, 'available': [p.id for p in available]}

    if not available:
        last_move['moved'] = None
    else:
        if pidx == 0:
            pid = get_pawn_id()  # get the pawn id sent by frontend
            
            log = "Human pid: " + str(pid) + "\nAvailable pawns: " + str([p.id for p in available])
            add_log(log)

            # If no pawn selected or invalid, tell frontend to retry
            if pid is None or pid not in [p.id for p in available]:
                add_log("Invalid or no pawn selected, waiting for valid selection...")
                return jsonify({
                    'game_id': gid,
                    'state': game.snapshot(),
                    'last_move': last_move,
                    'waiting_for_human': True,
                })

            pick_idx = next(i for i, p in enumerate(available) if p.id == pid)
            chosen = available[pick_idx]
        else:
            robot_idx = robot_choose(game.players, pidx, available)
            if (robot_idx < len(available) and robot_idx >= 0):
                chosen = available[robot_idx]
            else:
                chosen = available[0]
        
        last_move['before_board_idx'] = game.board.main_track.index(chosen) if chosen in game.board.main_track else None
        last_move['before_tile_position'] = chosen.tile_position

        log = "Player " + str(pidx) + " " + str(player.color.name) + " moves pawn " + str(chosen.id) + " from tile " + str(chosen.tile_position) + " with dice " + str(game.dice)
        add_log(log)
        move_pawn(chosen, player, game.dice, game.board)
        last_move['after_board_idx'] = game.board.main_track.index(chosen) if chosen in game.board.main_track else None
        last_move['after_tile_position'] = chosen.tile_position
        last_move['moved'] = chosen.id

    # increment and attach a move id so clients can identify new moves
    global move_counter
    move_counter += 1
    last_move['move_id'] = move_counter

    if game.dice != 6:
        game.current_idx = (game.current_idx + 1) % len(game.players)

    api_play_refresh(last_move)
    # push last_move payload to the stream so clients know a move occurred
    try:
        message_queue.put({'type': 'move', 'game_id': gid, 'last_move': last_move, 'move_id': move_counter})
    except Exception:
        # fallback to simple signal
        message_queue.put({'type': 'move', 'game_id': gid, 'last_move': last_move, 'move_id': move_counter})

    global move_idx
    move_idx = {'color': player.color.value, 'before_tile_position': last_move['before_tile_position'], 'after_tile_position': last_move['after_tile_position'], 'chosen': chosen.id, 'before_board_idx': last_move['before_board_idx'], 'after_board_idx': last_move['after_board_idx'], 'dice': last_move['dice']}

    return jsonify({'game_id': gid, 'state': game.snapshot(), 'last_move': last_move})

@app.route('/api/play/refresh', methods=['GET'])
def api_play_refresh(lastmove=None):
    global g_gid
    sm = GAME_FSMS.get(g_gid)

    if sm is None:
        return jsonify({'error': 'Game not started'}), 400

#    add_log(sm.game.snapshot())
    global move_idx
    global winner    
    return jsonify({'winner': winner, 'game_id': g_gid, 'state': sm.game.snapshot(), 'last_move': move_idx})

# This ensures the script runs only when executed directly
if __name__ == "__main__":
    # We'll run the app on port 8080 to avoid conflicts with Next.js's default port 3000
    app.run(debug=True, port=8080)