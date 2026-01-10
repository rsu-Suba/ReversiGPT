#ifndef MCTS_H
#define MCTS_H
#include "reversi_bitboard.h"
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class MCTSNode : public std::enable_shared_from_this<MCTSNode> {
public:
    ReversiBitboard game_board;
    int player;
    std::weak_ptr<MCTSNode> parent;
    int move;
    double prior_p;
    std::map<int, std::shared_ptr<MCTSNode>> children;
    int n_visits;
    double q_value;
    double sum_value;
    bool is_game_over = false;

    MCTSNode(ReversiBitboard board, int p, std::shared_ptr<MCTSNode> parent_node = nullptr, int m = -1, double prior = 0.0);

    double ucb_score(double c_puct) const;
    std::shared_ptr<MCTSNode> select_child(double c_puct);
    bool is_fully_expanded();
    void update(double value);
    std::vector<int> get_legal_moves();

private:
    std::vector<int> _legal_moves;
    bool _legal_moves_calculated = false;
};

class MCTS {
public:
    MCTS(py::object model, double c_puct = 1.41, int batch_size = 8);

    std::shared_ptr<MCTSNode> search(ReversiBitboard& board, int player, int num_simulations, bool add_noise);

private:
    py::object model;
    double c_puct;
    int batch_size;
    std::shared_ptr<MCTSNode> root;

    void expand_and_backup(const std::vector<std::shared_ptr<MCTSNode>>& search_path, const py::array_t<float>& policy_batch, const py::array_t<float>& value_batch);
    void batch_predict(const std::vector<std::shared_ptr<MCTSNode>>& leaf_nodes);
};

#endif