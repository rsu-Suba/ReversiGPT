import sys
import os
import random
import numpy as np
import tensorflow as tf
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AI.cpp.reversi_bitboard_cpp import ReversiBitboard
from AI.models.transformer import TokenAndPositionEmbedding, TransformerBlock
from AI.config import (
    R_SIMS_N,
    Model_Path,
    C_PUCT
)
from tensorflow.keras.optimizers.schedules import CosineDecay

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

MCTS_SIMS_PER_MOVE = R_SIMS_N
MODEL_PATH = Model_Path

@tf.keras.utils.register_keras_serializable()
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, warmup_steps, alpha=0.0):
        super(WarmupCosineDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.alpha = alpha
        self.cosine_decay = CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps - warmup_steps,
            alpha=alpha
        )

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_percent_done = step / self.warmup_steps
        warmup_learning_rate = self.initial_learning_rate * warmup_percent_done

        is_warmup = step < self.warmup_steps

        learning_rate = tf.cond(
            is_warmup,
            lambda: warmup_learning_rate,
            lambda: self.cosine_decay(step - self.warmup_steps)
        )
        return learning_rate

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "warmup_steps": self.warmup_steps,
            "alpha": self.alpha,
        }

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


def print_board(board_1d):
    print("  A B C D E F G H")
    for r in range(8):
        row_str = "";
        for c in range(8):
            piece = board_1d[r * 8 + c]
            if piece == 1:
                row_str += "🔴"
            elif piece == 2:
                row_str += "⚪️"
            else:
                row_str += "🟩"
        print(f"{r+1}{row_str}")

def get_human_move(legal_moves):
    while True:
        try:
            move_str = input("Your turn 🫵 (a1, pass): ").strip()
            if move_str.lower() == "pass":
                if -1 in legal_moves:
                    return -1
                else:
                    print("❌️Can't pass")
                    continue

            if len(move_str) != 2:
                raise ValueError("❌️Invalid move.")

            col_char = move_str[0].upper()
            row_char = move_str[1]

            if not ('A' <= col_char <= 'H' and '1' <= row_char <= '8'):
                raise ValueError("❌️Invalid move.")

            col = ord(col_char) - ord('A')
            row = int(row_char) - 1
            move = row * 8 + col

            if move not in legal_moves:
                print(f"❌️Invalid move.: {move_str}")
                print(f"Pleaceable: {[index_to_coord(m) for m in legal_moves if m != -1]}")
                continue
            return move
        except ValueError as e:
            print(f"Error: {e}, Enter again")
        except Exception as e:
            print(f"Unexpected error: {e}, Enter again")

def index_to_coord(index):
    if index == -1:
        return "PASS"
    row = index // 8
    col = index % 8
    return f"{chr(ord('A') + col)}{row + 1}"

def main():
    game_board = ReversiBitboard()

    human_player = random.choice([1, 2])
    ai_player = 3 - human_player

    human_color = "Black 🔴" if human_player == 1 else "White ⚪️"
    ai_color = "White ⚪️" if ai_player == 2 else "Black 🔴"
    print(f"You : {human_color}, AI : {ai_color}")

    current_player = 1

    try:
        with tf.keras.utils.custom_object_scope({'TokenAndPositionEmbedding': TokenAndPositionEmbedding, 'TransformerBlock': TransformerBlock, 'WarmupCosineDecay': WarmupCosineDecay}):
            model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded <- {MODEL_PATH}")
        mcts_ai = MCTS(model)
    except Exception as e:
        print(f"Error while loading model {e}")
        sys.exit(1)

    while not game_board.is_game_over():
        turn_player_name = "You" if current_player == human_player else "AI"
        turn_color = "Black" if current_player == 1 else "White"
        print(f"\n--- Now turn: {turn_color} ({turn_player_name}) ---")
        print_board(game_board.board_to_numpy())

        legal_moves = game_board.get_legal_moves()

        if not legal_moves:
            print(f"{turn_color} has no legal moves -> pass")
            game_board.apply_move(-1)
            current_player = 3 - current_player
            continue

        if current_player == human_player:
            move = get_human_move(legal_moves)
        else:
            print("AI 🤔🤔🤔...")
            search_result = mcts_ai.search(game_board, current_player, MCTS_SIMS_PER_MOVE)
            move = search_result[0] if search_result and search_result[0] is not None else -1
            print(f"AI 😓👍: {index_to_coord(move)}")

        game_board.apply_move(move)
        current_player = 3 - current_player

    print("\n--- Game finish ---")
    print_board(game_board.board_to_numpy())

    winner = game_board.get_winner()
    black_stones = game_board.count_set_bits(game_board.black_board)
    white_stones = game_board.count_set_bits(game_board.white_board)
    print(f"Scores - Black: {black_stones}, White: {white_stones}")

    if winner == 0:
        print("Draw")
    elif winner == human_player:
        print("You won")
    else:
        print("AI won")

if __name__ == "__main__":
    main()