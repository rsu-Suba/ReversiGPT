import numpy as np
import tensorflow as tf
import os
import math
import time
import sys
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AI.cpp.reversi_bitboard_cpp import ReversiBitboard
from AI.models.transformer import TokenAndPositionEmbedding, TransformerBlock, build_model
from AI.models.static_MoE import TokenAndPositionEmbedding, TransformerBlock, build_model, ExpertSearch, ExpertThink
from AI.models.dynamic_MoE import TokenAndPositionEmbedding, MHA, FFN, DynamicAssembly, build_model
from AI.training.train_loop import WarmupCosineDecay
from AI.config import (
    NUM_GAMES_COMPARE,
    COMPARE_SIMS_N,
    Model1_Path,
    Model2_Path,
    Model1_Name,
    Model2_Name
)

def _print_numpy_board(board_1d):
    print("  0 1 2 3 4 5 6 7")
    for r in range(8):
        row_str = f"{r}"
        for c in range(8):
            piece = board_1d[r * 8 + c]
            if piece == 1: row_str += "🔴"
            elif piece == 2: row_str += "⚪️"
            else: row_str += "🟩"
        print(row_str)

def board_to_input_planes_tf(board_1d_tf, current_player_tf):
    player_plane = tf.zeros((8, 8), dtype=tf.float32)
    opponent_plane = tf.zeros((8, 8), dtype=tf.float32)
    board_2d_tf = tf.reshape(board_1d_tf, (8, 8))
    current_player_mask = tf.cast(tf.equal(board_2d_tf, current_player_tf), tf.float32)
    opponent_player_mask = tf.cast(tf.equal(board_2d_tf, 3 - current_player_tf), tf.float32)
    player_plane += current_player_mask
    opponent_plane += opponent_player_mask
    return tf.stack([player_plane, opponent_plane], axis=-1)

class MCTSNode:
    def __init__(self, game_board: ReversiBitboard, player, parent=None, move=None, prior_p=0.0):
        self.game_board = game_board
        self.player = player
        self.parent = parent
        self.move = move
        self.prior_p = prior_p
        self.children = {}
        self.n_visits = 0
        self.q_value = 0.0

    def ucb_score(self, c_puct):
        if self.parent is None: return self.q_value
        return -self.q_value + c_puct * self.prior_p * math.sqrt(self.parent.n_visits) / (1 + self.n_visits)

    def select_child(self, c_puct):
        return max(self.children.values(), key=lambda child: child.ucb_score(c_puct))

    def is_fully_expanded(self):
        return len(self.children) == len(self.game_board.get_legal_moves())

    def update(self, value):
        self.n_visits += 1
        self.q_value += (value - self.q_value) / self.n_visits

class MCTS:
    def __init__(self, model, c_puct=1.41):
        self.model = model
        self.c_puct = c_puct
        self._predict_graph = tf.function(
            self._predict_internal,
            input_signature=[
                tf.TensorSpec(shape=[64], dtype=tf.int8),
                tf.TensorSpec(shape=(), dtype=tf.int8)
            ]
        )

    def _predict_internal(self, board_tensor, player_tensor):
        input_planes = board_to_input_planes_tf(board_tensor, player_tensor)
        input_tensor_batch = tf.expand_dims(input_planes, axis=0)
        policy, value = self.model(input_tensor_batch, training=False)
        return policy[0], value[0][0]

    def _predict(self, board, player):
        board_tensor = tf.convert_to_tensor(board, dtype=tf.int8)
        player_tensor = tf.convert_to_tensor(player, dtype=tf.int8)
        policy, value = self._predict_graph(board_tensor, player_tensor)
        return policy.numpy(), value.numpy()

    def search(self, game_board: ReversiBitboard, player, num_simulations):
        self.root = MCTSNode(game_board, player)
        for _ in range(num_simulations):
            node = self.root
            sim_game_board = ReversiBitboard()
            sim_game_board.black_board = game_board.black_board
            sim_game_board.white_board = game_board.white_board
            sim_game_board.current_player = game_board.current_player
            sim_game_board.passed_last_turn = game_board.passed_last_turn
            sim_player = player
            path = [node]

            while node.is_fully_expanded() and node.children and not sim_game_board.is_game_over():
                selected_child = node.select_child(self.c_puct)
                sim_game_board.apply_move(selected_child.move)
                sim_player = sim_game_board.current_player
                node = selected_child
                path.append(node)

            value = 0

            if not sim_game_board.is_game_over():
                valid_moves = sim_game_board.get_legal_moves()
                if valid_moves:
                    policy, value = self._predict(sim_game_board.board_to_numpy(), sim_player)
                    masked_policy = {move: policy[move] for move in valid_moves}
                    policy_sum = sum(masked_policy.values())
                    if policy_sum > 0:
                        masked_policy = {move: p / policy_sum for move, p in masked_policy.items()}
                    else:
                        masked_policy = {move: 1.0 / len(valid_moves) for move in valid_moves}

                    for move, prior in masked_policy.items():
                        if move not in node.children:
                            new_game_board = ReversiBitboard()
                            new_game_board.black_board = sim_game_board.black_board
                            new_game_board.white_board = sim_game_board.white_board
                            new_game_board.current_player = sim_game_board.current_player
                            new_game_board.passed_last_turn = sim_game_board.passed_last_turn
                            new_game_board.apply_move(move)
                            new_player = new_game_board.current_player
                            node.children[move] = MCTSNode(new_game_board, new_player, parent=node, move=move, prior_p=prior)
                else:
                    pass_player = 3 - sim_player
                    if sim_game_board.get_legal_moves():
                        _, value = self._predict(sim_game_board.board_to_numpy(), pass_player)
                        value = -value
                    else:
                        winner = sim_game_board.get_winner()
                        value = 0 if winner == 0 else (1 if winner == sim_player else -1)
            else:
                winner = sim_game_board.get_winner()
                value = 0 if winner == 0 else (1 if winner == sim_player else -1)

            for node_in_path in reversed(path):
                perspective_value = value if node_in_path.player == sim_player else -value
                node_in_path.update(perspective_value)

        if not self.root.children:
            return None
        return self.root

class MCTS_AIPlayer:
    def __init__(self, model_path, name, sims_per_move):
        with tf.keras.utils.custom_object_scope({'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
                                                 'TransformerBlock': TransformerBlock,
                                                 'WarmupCosineDecay': WarmupCosineDecay, 
                                                 'DynamicAssembly': DynamicAssembly}):
            self.model = tf.keras.models.load_model(model_path)
        self.mcts = MCTS(self.model)
        self.name = name
        self.sims_per_move = sims_per_move
        print(f"Initialized AI '{name}'. Model: {model_path}, SimsN: {sims_per_move}")

    def choose_move(self, game_board: ReversiBitboard, player, verbose=False, temperature=1.0):
        root_node = self.mcts.search(game_board, player, self.sims_per_move)

        if root_node is None or not root_node.children:
            return None, 0, 0.0

        moves = list(root_node.children.keys())
        visit_counts = np.array([root_node.children[move].n_visits for move in moves])

        if temperature == 0:
            best_move_idx = np.argmax(visit_counts)
            chosen_move = moves[best_move_idx]
        else:
            scaled_visits = visit_counts**(1.0 / temperature)
            probabilities = scaled_visits / np.sum(scaled_visits)
            chosen_move = np.random.choice(moves, p=probabilities)

        chosen_node = root_node.children[chosen_move]
        visits = chosen_node.n_visits
        q_value = -chosen_node.q_value

        return chosen_move, visits, q_value

def simulate_game(player1_ai, player2_ai, verbose=False, black_thinks_like_white=False):
    game_board = ReversiBitboard()
    current_player = 1

    player1_q_values = []
    player2_q_values = []

    if verbose:
        print("\n--- New game ---")
    while not game_board.is_game_over():
        current_ai = player1_ai if current_player == 1 else player2_ai

        if not game_board.get_legal_moves():
            if verbose: print(f"Player {current_player} ({current_ai.name}) has no legal move -> Passed")
            game_board.apply_move(-1)
            current_player = 3 - current_player
            continue

        temp_game_board = ReversiBitboard()
        temp_game_board.black_board = game_board.black_board
        temp_game_board.white_board = game_board.white_board
        temp_game_board.current_player = game_board.current_player
        temp_game_board.passed_last_turn = game_board.passed_last_turn

        search_player = current_player

        if black_thinks_like_white and current_player == 1:
            temp_black_board = temp_game_board.black_board
            temp_white_board = temp_game_board.white_board
            temp_game_board.black_board = temp_white_board
            temp_game_board.white_board = temp_black_board
            search_player = 2

        num_turns = len(game_board.history)
        temperature = 1.0 if num_turns < 30 else 0.0
        move, visits, q_value = current_ai.choose_move(temp_game_board, search_player, verbose=verbose, temperature=temperature)

        if current_ai == player1_ai:
            player1_q_values.append(q_value)
        else:
            player2_q_values.append(q_value)

        if move is None:
            if verbose: print(f"AI {current_ai.name} passed")
            game_board.apply_move(-1)
            current_player = 3 - current_player
            continue

        if verbose:
            print(f"{current_player} ({current_ai.name}) chose: {move} (Visits: {visits}, Q num: {q_value:.4f})")
        game_board.apply_move(move)
        current_player = 3 - current_player

    winner = game_board.get_winner()
    black_count = game_board.count_set_bits(game_board.black_board)
    white_count = game_board.count_set_bits(game_board.white_board)

    if verbose:
        print(f"\nGame finish. Scores: Black = {black_count}, White = {white_count}")
        _print_numpy_board(game_board.board_to_numpy())
        if winner == 1: print(f"Win: Black ({player1_ai.name})")
        elif winner == 2: print(f"Win: White ({player2_ai.name})")
        else: print("Draw")

    return winner, black_count, white_count, player1_q_values, player2_q_values

def run_comparison(model1_path, model1_name, model2_path, model2_name, num_games, SIMS_N, game_verbose=False, black_thinks_like_white=False):
    if not os.path.exists(model1_path):
        print(f"Model 404 -> {model1_path}")
        return "Error"
    if not os.path.exists(model2_path):
        print(f"Model 404 -> {model2_path}")
        return "Error"

    print(f"--- Model compare: {model1_name} vs {model2_name} ---")
    ai1 = MCTS_AIPlayer(model1_path, model1_name, SIMS_N)
    ai2 = MCTS_AIPlayer(model2_path, model2_name, SIMS_N)

    wins = {ai1.name: 0, ai2.name: 0, "Draw": 0}
    total_stones = {ai1.name: 0, ai2.name: 0}
    all_q_values_ai1 = []
    all_q_values_ai2 = []
    start_time = time.time()

    print(f"\nStart simulation -> {num_games}")
    for i in range(num_games):
        if i % 2 == 0:
            p1, p2 = ai1, ai2
            p1_name, p2_name = ai1.name, ai2.name
        else:
            p1, p2 = ai2, ai1
            p1_name, p2_name = ai2.name, ai1.name

        winner, black_count, white_count, p1_q_values, p2_q_values = simulate_game(p1, p2, verbose=game_verbose)

        if p1 == ai1:
            all_q_values_ai1.extend(p1_q_values)
            all_q_values_ai2.extend(p2_q_values)
        else:
            all_q_values_ai1.extend(p2_q_values)
            all_q_values_ai2.extend(p1_q_values)

        if winner == 1:
            wins[p1_name] += 1
            total_stones[p1_name] += black_count
            total_stones[p2_name] += white_count
        elif winner == 2:
            wins[p2_name] += 1
            total_stones[p1_name] += black_count
            total_stones[p2_name] += white_count
        else:
            wins["Draw"] += 1
            total_stones[p1_name] += black_count
            total_stones[p2_name] += white_count

        progress = (i + 1) / num_games
        bar = '#' * int(progress * 20)
        print(f"[{bar:<20}] {int(progress*100)}% | Wins: {ai1.name} {wins[ai1.name]} - {wins[ai2.name]} {ai2.name} (Draw: {wins['Draw']})", end='\r')

    end_time = time.time()
    print("\n\nSimulation finish")
    print(f"Total: {end_time - start_time:.2f} sec")
    print("\n--- Results ---")
    print(f"SimsN : {SIMS_N}")
    print(f"{ai1.name} wins: {wins[ai1.name]}")
    print(f"{ai2.name} wins: {wins[ai2.name]}")
    print(f"Draw: {wins['Draw']}")
    print("-----------------")
    avg_stones_1 = total_stones[ai1.name] / num_games
    avg_stones_2 = total_stones[ai2.name] / num_games
    print(f"{ai1.name} ave scores: {avg_stones_1:.1f}")
    print(f"{ai2.name} ave scores: {avg_stones_2:.1f}")

    if all_q_values_ai1:
        print(f"{ai1.name} ave Q num: {np.mean(all_q_values_ai1):.4f}")
    else:
        print(f"No Q num data of {ai1.name}")
    if all_q_values_ai2:
        print(f"{ai2.name} ave Q num: {np.mean(all_q_values_ai2):.4f}")
    else:
        print(f"No Q num data of {ai2.name}")
    print("-----------------")

    if wins[ai1.name] > wins[ai2.name]:
        return ai1.name
    elif wins[ai2.name] > wins[ai1.name]:
        return ai2.name
    else:
        return "Draw"

if __name__ == "__main__":

    winner_name = run_comparison(Model1_Path, Model1_Name, Model2_Path, Model2_Name, NUM_GAMES_COMPARE, COMPARE_SIMS_N, game_verbose=False, black_thinks_like_white=True)
    print(f"Winner: {winner_name}")