import asyncio
import json
import random
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
import os

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from AI.models.transformer_model import TokenAndPositionEmbedding, TransformerBlock
from AI.config import R_SIMS_N, C_PUCT, Model_Path
from AI.cpp.reversi_bitboard_cpp import ReversiBitboard
from review import MCTS, RandomAI
app = FastAPI()
html_path = os.path.join(os.path.dirname(__file__), "web_ui/review_client/index.html")
with open(html_path, "r") as f:
    html_content = f.read()

@app.get("/")
async def get():
    return HTMLResponse(html_content)

def load_ai_model():
    """Loads the Keras model with custom objects."""
    try:
        with tf.keras.utils.custom_object_scope({'TokenAndPositionEmbedding': TokenAndPositionEmbedding, 'TransformerBlock': TransformerBlock}):
            model = tf.keras.models.load_model(Model_Path, compile=False)
        print(f"Model loaded <- {Model_Path}")
        return model
    except Exception as e:
        print(f"Error while loading model: {e}")
        return None

def get_policy_heatmap(mcts_root, board_size=64):
    """Extracts policy/visit counts from MCTS root to create a heatmap."""
    heatmap = np.zeros(board_size, dtype=float)
    if mcts_root and mcts_root.children:
        total_visits = sum(child.n_visits for child in mcts_root.children.values())
        if total_visits > 0:
            for move, child in mcts_root.children.items():
                if move >= 0 and move < board_size:
                    heatmap[move] = child.n_visits / total_visits
    return heatmap.tolist()

async def run_game(websocket: WebSocket, mcts_ai: MCTS, random_ai: RandomAI, board_id: int):
    """Runs a SINGLE AI vs AI game and sends updates for a specific board."""
    game_board = ReversiBitboard()
    current_player = 1
    ai1_is_mcts = random.choice([True, False])

    while not game_board.is_game_over():
        is_mcts_turn = (current_player == 1 and ai1_is_mcts) or \
                       (current_player == 2 and not ai1_is_mcts)
        
        legal_moves = game_board.get_legal_moves()
        policy_heatmap = []
        flipped_indices = []

        if not legal_moves:
            game_board.apply_move(-1)
            current_player = 3 - current_player
            continue

        payload_data = {
            "board": game_board.board_to_numpy().tolist(),
            "legal_moves": legal_moves,
        }

        if is_mcts_turn:
            search_board = ReversiBitboard()
            search_board.black_board = game_board.black_board
            search_board.white_board = game_board.white_board
            search_board.current_player = game_board.current_player
            search_board.passed_last_turn = game_board.passed_last_turn
            search_player = game_board.current_player

            if search_player == 1:
                search_board.black_board, search_board.white_board = search_board.white_board, search_board.black_board
                search_board.current_player = 2
                search_player = 2

            mcts_ai.search(search_board, search_player, R_SIMS_N)
            move = max(mcts_ai.root.children.keys(), key=lambda m: mcts_ai.root.children[m].n_visits)
            
            payload_data["policy_heatmap"] = get_policy_heatmap(mcts_ai.root)
            payload_data["flipped_indices"] = search_board.get_flipped_indices(move)
        else:
            move = random_ai.get_move(game_board, current_player)
            payload_data["policy_heatmap"] = []
            payload_data["flipped_indices"] = []

        await websocket.send_json({
            "type": "board_update",
            "board_id": board_id,
            "payload": payload_data,
        })
        
        await asyncio.sleep(0.001)

        game_board.apply_move(move)
        current_player = 3 - current_player

    await websocket.send_json({
        "type": "board_update",
        "board_id": board_id,
        "payload": {
            "board": game_board.board_to_numpy().tolist(),
            "legal_moves": [], "policy_heatmap": [], "flipped_indices": []
        },
    })

async def run_game_loop(websocket: WebSocket, board_id: int, mcts_ai: MCTS, random_ai: RandomAI):
    """Runs games for a single board_id back-to-back until cancelled."""
    while True:
        try:
            await run_game(websocket, mcts_ai, random_ai, board_id)
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            print(f"Game loop for board {board_id} was cancelled.")
            break
        except Exception as e:
            print(f"Error in game loop for board {board_id}: {e}")
            break

client_tasks = {}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    print(f"Client {client_id} connected.")
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "start":
                num_boards = message.get("num_boards", 1)
                if client_id in client_tasks:
                    for old_task in client_tasks[client_id]:
                        old_task.cancel()
                
                print(f"Client {client_id} starting {num_boards} game loops.")
                model = load_ai_model()
                if model is None:
                    break
                mcts_ai = MCTS(model)
                random_ai = RandomAI()
                
                tasks = [
                    asyncio.create_task(run_game_loop(websocket, i, mcts_ai, random_ai))
                    for i in range(num_boards)
                ]
                client_tasks[client_id] = tasks

            elif message.get("type") == "stop":
                if client_id in client_tasks:
                    print(f"Client {client_id} stopping game loops.")
                    for task in client_tasks[client_id]:
                        task.cancel()
                    client_tasks.pop(client_id, None)

    except WebSocketDisconnect:
        print(f"Client {client_id} disconnected.")
    finally:
        tasks = client_tasks.pop(client_id, [])
        for task in tasks:
            task.cancel()
        print(f"Cleaned up for client {client_id}.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
