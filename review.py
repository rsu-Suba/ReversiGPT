import numpy as np
import random
import tensorflow as tf
from Database.transformer_Model import TokenAndPositionEmbedding, TransformerBlock
from config import R_SIMS_N, C_PUCT, Model_Path, Play_Games_Num
import math
from reversi_bitboard_cpp import ReversiBitboard

NUM_GAMES_TO_PLAY = Play_Games_Num
MCTS_SIMS_PER_MOVE = R_SIMS_N
MODEL_PATH = Model_Path

def board_to_input_planes_tf(board_1d_tf, current_player_tf):
    player_plane = tf.zeros((8, 8), dtype=tf.float32)
    opponent_plane = tf.zeros((8, 8), dtype=tf.float32)
    board_2d_tf = tf.reshape(board_1d_tf, (8, 8))
    current_player_mask = tf.cast(tf.equal(board_2d_tf, current_player_tf), tf.float32)
    opponent_player_mask = tf.cast(tf.equal(board_2d_tf, 3 - current_player_tf), tf.float32)
    player_plane += current_player_mask
    opponent_plane += opponent_player_mask
    return tf.stack([player_plane, opponent_plane], axis=-1)

def print_board_from_numpy(board_1d):
    print("  0 1 2 3 4 5 6 7")
    for r in range(8):
        row_str = f"{r}"
        for c in range(8):
            piece = board_1d[r * 8 + c]
            if piece == 1: row_str += "🔴"
            elif piece == 2: row_str += "⚪️"
            else: row_str += "🟩"
        print(row_str)

class RandomAI:
    def get_move(self, game_board: ReversiBitboard, player):
        valid_moves = game_board.get_legal_moves()
        if not valid_moves:
            return None
        return random.choice(valid_moves)

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
    def __init__(self, model, c_puct=C_PUCT):
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
            return None, 0, 0

        best_move = max(self.root.children.keys(), key=lambda move: self.root.children[move].n_visits)
        return best_move, self.root.children[best_move].n_visits, self.root.children[best_move].q_value

def play_game(mcts_ai, random_ai, mcts_player_is_black):
    game_board = ReversiBitboard()
    current_player = 1
    mcts_q_values = []

    while not game_board.is_game_over():
        is_mcts_turn = (current_player == 1 and mcts_player_is_black) or (current_player == 2 and not mcts_player_is_black)

        valid_moves = game_board.get_legal_moves()
        if not valid_moves:
            game_board.apply_move(-1)
            current_player = 3 - current_player
            continue

        if is_mcts_turn:
            search_result = mcts_ai.search(game_board, current_player, MCTS_SIMS_PER_MOVE)
            move = search_result[0] if search_result and search_result[0] is not None else -1
            q_value = search_result[2] if search_result and search_result[2] is not None else 0.0
            mcts_q_values.append(q_value)
        else:
            move = random_ai.get_move(game_board, current_player)

        if move is None or move == -1:
            game_board.apply_move(-1)
            current_player = 3 - current_player
            continue

        game_board.apply_move(move)
        current_player = 3 - current_player

    winner = game_board.get_winner()
    if winner == 0:
        result = "draw"
    elif (winner == 1 and mcts_player_is_black) or (winner == 2 and not mcts_player_is_black):
        result = "mcts_win"
    else:
        result = "random_win"
    return result, game_board, mcts_q_values


if __name__ == "__main__":
    print("--- AI vs Random bot ---")
    try:
        with tf.keras.utils.custom_object_scope({'TokenAndPositionEmbedding': TokenAndPositionEmbedding, 'TransformerBlock': TransformerBlock}):
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"Model loaded <- {MODEL_PATH}")
        mcts_ai = MCTS(model)
    except Exception as e:
        print(f"Error while loading model: {e}")
        exit()
    random_ai = RandomAI()

    mcts_wins = 0
    random_wins = 0
    draws = 0
    total_mcts_stones = 0
    total_random_stones = 0
    all_mcts_q_values = []

    for i in range(NUM_GAMES_TO_PLAY):
        mcts_is_black = random.choice([True, False])
        print(f"\n--- Game {i+1}/{NUM_GAMES_TO_PLAY} | AI: {'Black' if mcts_is_black else 'White'} ---")
        result, final_board, mcts_q_values = play_game(mcts_ai, random_ai, mcts_is_black)
        all_mcts_q_values.extend(mcts_q_values)

        black_stones = final_board.count_set_bits(final_board.black_board)
        white_stones = final_board.count_set_bits(final_board.white_board)

        if mcts_is_black:
            mcts_score = black_stones
            random_score = white_stones
        else:
            mcts_score = white_stones
            random_score = black_stones

        total_mcts_stones += mcts_score
        total_random_stones += random_score

        if result == "mcts_win":
            mcts_wins += 1
            print(f"Game {i+1} result: AI Win")
        elif result == "random_win":
            random_wins += 1
            print(f"Game {i+1} result: Random bot Win")
            final_board_numpy = final_board.board_to_numpy()
            print_board_from_numpy(final_board_numpy)
        else:
            draws += 1
            print(f"Game {i+1} result: Draw")

        print(f"Scores - Black: {black_stones}, White: {white_stones}")

    print("\n--- Result ---")
    print(f"Games: {NUM_GAMES_TO_PLAY}")
    print(f"AI wins: {mcts_wins} ({((mcts_wins / NUM_GAMES_TO_PLAY) * 100):.2f}%) ")
    print(f"Bot wins: {random_wins}")
    print(f"Draws: {draws}")
    print(f"AI average stones: {total_mcts_stones / NUM_GAMES_TO_PLAY:.2f}")
    print(f"Random bot average stones: {total_random_stones / NUM_GAMES_TO_PLAY:.2f}")
    if all_mcts_q_values:
        print(f"AI average Q value: {np.mean(all_mcts_q_values):.4f}")
    else:
        print("No Q value data for AI.")