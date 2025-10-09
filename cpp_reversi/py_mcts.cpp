#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "mcts.h"

namespace py = pybind11;

PYBIND11_MODULE(reversi_mcts_cpp, m) {
    m.doc() = "MCTS module implemented in C++";

    py::class_<MCTSNode, std::shared_ptr<MCTSNode>>(m, "MCTSNode")
        .def(py::init<ReversiBitboard, int, std::shared_ptr<MCTSNode>, int, double>(), 
             py::arg("board"), py::arg("player"), py::arg("parent") = nullptr, 
             py::arg("move") = -1, py::arg("prior_p") = 0.0)
        .def_readonly("player", &MCTSNode::player)
        .def_readonly("move", &MCTSNode::move)
        .def_readonly("n_visits", &MCTSNode::n_visits)
        .def_readonly("q_value", &MCTSNode::q_value)
        .def("get_legal_moves", &MCTSNode::get_legal_moves)
        .def_property_readonly("children", [](MCTSNode &self) {
            py::dict children_dict;
            for (auto const& [move, child_ptr] : self.children) {
                children_dict[py::cast(move)] = py::cast(child_ptr);
            }
            return children_dict;
        });

    py::class_<MCTS, std::shared_ptr<MCTS>>(m, "MCTS")
        .def(py::init<py::object, double, int>(), py::arg("model"), py::arg("c_puct") = 1.41, py::arg("batch_size") = 8)
        .def("search", &MCTS::search, py::arg("board"), py::arg("player"), py::arg("num_simulations"), py::arg("add_noise") = false,
            py::return_value_policy::reference_internal);
}