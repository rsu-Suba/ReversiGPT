#include "mcts.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>

MCTSNode::MCTSNode(ReversiBitboard board, int p, std::shared_ptr<MCTSNode> parent_node, int m, double prior)
    : game_board(board), player(p), parent(parent_node), move(m), prior_p(prior), 
      n_visits(0), q_value(0.0), sum_value(0.0), _legal_moves_calculated(false) {
    is_game_over = game_board.is_game_over();
}

double MCTSNode::ucb_score(double c_puct) const {
    auto parent_shared = parent.lock();
    if (!parent_shared) return 0.0;

    double u = c_puct * prior_p * std::sqrt((double)parent_shared->n_visits) / (1 + n_visits);

    return -q_value + u;
}

std::shared_ptr<MCTSNode> MCTSNode::select_child(double c_puct) {
    std::shared_ptr<MCTSNode> best_child = nullptr;
    double max_score = -std::numeric_limits<double>::infinity();

    for (auto const& [m, child] : children) {
        double score = child->ucb_score(c_puct);
        if (score > max_score) {
            max_score = score;
            best_child = child;
        }
    }
    return best_child;
}

bool MCTSNode::is_fully_expanded() {
    return !children.empty();
}

void MCTSNode::update(double value) {
    n_visits++;
    sum_value += value;
    q_value = sum_value / n_visits;
    
    auto parent_shared = parent.lock();
    if (parent_shared) {
        parent_shared->update(-value);
    }
}

std::vector<int> MCTSNode::get_legal_moves() {
    if (!_legal_moves_calculated) {
        _legal_moves = game_board.get_legal_moves();
        _legal_moves_calculated = true;
    }
    return _legal_moves;
}

MCTS::MCTS(py::object model, double c_puct, int batch_size)
    : model(model), c_puct(c_puct), batch_size(batch_size) {}

std::shared_ptr<MCTSNode> MCTS::search(ReversiBitboard& board, int player, int num_simulations, bool add_noise) {
    root = std::make_shared<MCTSNode>(board, player);
    std::vector<std::shared_ptr<MCTSNode>> root_leaves;
    root_leaves.push_back(root);
    batch_predict(root_leaves);

    int sims_done = 0;
    while (sims_done < num_simulations) {
        std::vector<std::shared_ptr<MCTSNode>> batch_leaves;
        std::vector<std::vector<std::shared_ptr<MCTSNode>>> paths;

        for (int b = 0; b < batch_size && sims_done < num_simulations; ++b) {
            auto node = root;
            std::vector<std::shared_ptr<MCTSNode>> path;
            path.push_back(node);

            while (node->is_fully_expanded() && !node->is_game_over) {
                node = node->select_child(c_puct);
                path.push_back(node);
            }

            if (node->is_game_over) {
                int winner = node->game_board.get_winner();
                double value = 0.0;
                if (winner != 0) {
                    value = (winner == node->player) ? 1.0 : -1.0;
                }
                for (int i = path.size() - 1; i >= 1; --i) {
                    path[i-1]->update(-value);
                }
                node->n_visits++; 
                sims_done++;
                continue;
            }

            batch_leaves.push_back(node);
            sims_done++;
        }

        if (!batch_leaves.empty()) {
            batch_predict(batch_leaves);
        }
    }
    
    return root;
}

#include <cstring>

void MCTS::batch_predict(const std::vector<std::shared_ptr<MCTSNode>>& leaf_nodes) {
    if (leaf_nodes.empty()) return;

    size_t batch_len = leaf_nodes.size();
    py::array_t<int8_t> board_batch({(ssize_t)batch_len, (ssize_t)64});
    auto buf = board_batch.request();
    int8_t* ptr = (int8_t*)buf.ptr;

    py::list player_batch;

    for (size_t i = 0; i < batch_len; ++i) {
        std::vector<int8_t> vec = leaf_nodes[i]->game_board.board_to_numpy();
        std::memcpy(ptr + i * 64, vec.data(), 64 * sizeof(int8_t));
        
        player_batch.append(leaf_nodes[i]->player);
    }

    try {
        py::tuple result = model.attr("_predict_internal_cpp")(board_batch, player_batch);
        
        py::array_t<float> policy_batch = result[0].cast<py::array_t<float>>();
        py::array_t<float> value_batch = result[1].cast<py::array_t<float>>();

        expand_and_backup(leaf_nodes, policy_batch, value_batch);
    } catch (py::error_already_set& e) {
        std::cerr << "MCTS C++ error calling model: " << e.what() << std::endl;
    }
}

void MCTS::expand_and_backup(const std::vector<std::shared_ptr<MCTSNode>>& leaf_nodes, 
                             const py::array_t<float>& policy_batch, 
                             const py::array_t<float>& value_batch) {
    auto p_buf = policy_batch.request();
    auto v_buf = value_batch.request();
    float* p_ptr = (float*)p_buf.ptr;
    float* v_ptr = (float*)v_buf.ptr;

    for (size_t i = 0; i < leaf_nodes.size(); ++i) {
        auto node = leaf_nodes[i];
        double value = (double)v_ptr[i];
        std::vector<int> legal_moves = node->get_legal_moves();
        if (!legal_moves.empty()) {
            double sum_exp = 0.0;
            std::vector<double> filtered_priors;
            for (int m : legal_moves) {
                double p = (double)p_ptr[i * 64 + m];
                filtered_priors.push_back(p);
                sum_exp += p;
            }

            for (size_t j = 0; j < legal_moves.size(); ++j) {
                int m = legal_moves[j];
                double prior = (sum_exp > 0) ? filtered_priors[j] / sum_exp : 1.0 / legal_moves.size();
                
                ReversiBitboard next_board = node->game_board;
                next_board.apply_move(m);
                int next_player = next_board.current_player;
                
                node->children[m] = std::make_shared<MCTSNode>(next_board, next_player, node, m, prior);
            }
        }

        node->update(value);
    }
}
