#include "mcts.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>
#include <iostream>
#include <pybind11/stl.h>

namespace py = pybind11;

// MCTSNode Implementation
MCTSNode::MCTSNode(ReversiBitboard board, int p, std::shared_ptr<MCTSNode> parent_node, int m, double prior)
    : game_board(board), player(p), parent(parent_node), move(m), prior_p(prior), 
      n_visits(0), q_value(0.0), sum_value(0.0) {
    is_game_over = game_board.is_game_over();
}

double MCTSNode::ucb_score(double c_puct) const {
    if (n_visits == 0) {
        return std::numeric_limits<double>::infinity();
    }
    if (auto p = parent.lock()) {
        return -q_value + c_puct * prior_p * std::sqrt(static_cast<double>(p->n_visits)) / (1 + n_visits);
    }
    return -q_value;
}

std::shared_ptr<MCTSNode> MCTSNode::select_child(double c_puct) {
    std::shared_ptr<MCTSNode> best_child = nullptr;
    double max_score = -std::numeric_limits<double>::infinity();
    for (auto const& [move, child] : children) {
        double score = child->ucb_score(c_puct);
        if (score > max_score) {
            max_score = score;
            best_child = child;
        }
    }
    return best_child;
}

std::vector<int> MCTSNode::get_legal_moves() {
    if (!_legal_moves_calculated) {
        _legal_moves = game_board.get_legal_moves();
        _legal_moves_calculated = true;
    }
    return _legal_moves;
}

bool MCTSNode::is_fully_expanded() {
    return children.size() > 0 && children.size() == get_legal_moves().size();
}

void MCTSNode::update(double value) {
    n_visits++;
    sum_value += value;
    q_value = sum_value / n_visits;
}

// MCTS Implementation
MCTS::MCTS(py::object model, double c_puct, int batch_size) 
    : model(model), c_puct(c_puct), batch_size(batch_size), root(nullptr) {}

void MCTS::batch_predict(const std::vector<std::shared_ptr<MCTSNode>>& leaf_nodes) {
    if (leaf_nodes.empty()) return;

    py::gil_scoped_acquire acquire;
    // Prepare buffer for board data
    std::vector<int8_t> board_data_buffer;
    board_data_buffer.reserve(leaf_nodes.size() * 64); // 64 elements per board

    for (const auto& node : leaf_nodes) {
        // Get the numpy representation of the board
        std::vector<int8_t> numpy_board_vec = node->game_board.board_to_numpy();
        // Append it to the buffer
        board_data_buffer.insert(board_data_buffer.end(), numpy_board_vec.begin(), numpy_board_vec.end());
    }

    // Create a py::array_t from the buffer with the correct shape
    // The shape should be (number_of_leaf_nodes, 64)
    std::vector<py::ssize_t> shape = {(py::ssize_t)leaf_nodes.size(), 64};
    py::array_t<int8_t> board_batch(
        shape,                      // Shape
        board_data_buffer.data()    // Pointer to data
    );
    
    std::vector<int> player_batch;
    for(const auto& node : leaf_nodes) {
        player_batch.push_back(node->player);
    }

    py::tuple result = model.attr("_predict_internal_cpp")(board_batch, py::cast(player_batch));
    py::array_t<float> policy_batch = result[0].cast<py::array_t<float>>();
    py::array_t<float> value_batch_py = result[1].cast<py::array_t<float>>();
    auto value_batch_unchecked = value_batch_py.unchecked<1>();
    auto policy_proxy = policy_batch.unchecked<2>();

    std::vector<float> value_batch(leaf_nodes.size());
    for(size_t i = 0; i < leaf_nodes.size(); ++i) {
        value_batch[i] = value_batch_unchecked(i);
    }

    py::gil_scoped_release release;

    for (size_t i = 0; i < leaf_nodes.size(); ++i) {
        auto node = leaf_nodes[i];
        double value = value_batch[i];

        std::vector<int> valid_moves = node->get_legal_moves();
        if (!valid_moves.empty()) {
            std::map<int, float> policy_map;
            float sum_policy = 0.0f;
            for (int move : valid_moves) {
                float p = policy_proxy(i, move);
                policy_map[move] = p;
                sum_policy += p;
            }
            if (sum_policy <= 1e-9) sum_policy = 1.0;

            for (int move : valid_moves) {
                if (node->children.find(move) == node->children.end()) {
                    ReversiBitboard new_board = node->game_board;
                    new_board.apply_move(move);
                    double prior = policy_map[move] / sum_policy;
                    node->children[move] = std::make_shared<MCTSNode>(new_board, new_board.current_player, node, move, prior);
                }
            }
        }

        // Backup
        double current_value = value;
        std::shared_ptr<MCTSNode> temp_node = node;
        while(temp_node != nullptr) {
            temp_node->update(current_value);
            current_value = -current_value;
            if(auto p = temp_node->parent.lock()) {
                temp_node = p;
            } else {
                break;
            }
        }
    }
}

std::shared_ptr<MCTSNode> MCTS::search(ReversiBitboard& board, int player, int num_simulations, bool add_noise) {
    if (root == nullptr || root->game_board.black_board != board.black_board || root->game_board.white_board != board.white_board) {
        root = std::make_shared<MCTSNode>(board, player);
    }

    if (add_noise) {
        // This part is simplified. A full implementation would apply noise to the root's priors after a first prediction.
    }

    std::vector<std::shared_ptr<MCTSNode>> leaf_nodes;
    for (int i = 0; i < num_simulations; ++i) {
        std::shared_ptr<MCTSNode> node = root;

        while (node->is_fully_expanded() && !node->is_game_over) {
            node = node->select_child(c_puct);
        }

        if (node->is_game_over) {
            int winner = node->game_board.get_winner();
            double value = 0.0;
            if (winner != 0) {
                value = (winner == node->player) ? 1.0 : -1.0;
            }

            std::shared_ptr<MCTSNode> temp_node = node;
            while(temp_node != nullptr) {
                temp_node->update(value);
                value = -value;
                if(auto p = temp_node->parent.lock()) {
                    temp_node = p;
                } else {
                    break;
                }
            }
            continue;
        }

        leaf_nodes.push_back(node);

        if (leaf_nodes.size() >= static_cast<size_t>(batch_size)) {
            batch_predict(leaf_nodes);
            leaf_nodes.clear();
        }
    }
    if (!leaf_nodes.empty()) {
        batch_predict(leaf_nodes);
    }

    return root;
}