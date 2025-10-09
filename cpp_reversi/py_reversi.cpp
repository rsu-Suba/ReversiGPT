#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "reversi_bitboard.h"

namespace py = pybind11;

int transform_horizontal_py(int move_idx) {
    int row = move_idx / 8;
    int col = move_idx % 8;
    return row * 8 + (7 - col);
}

int transform_vertical_py(int move_idx) {
    int row = move_idx / 8;
    int col = move_idx % 8;
    return (7 - row) * 8 + col;
}

int transform_transpose_main_py(int move_idx) {
    int row = move_idx / 8;
    int col = move_idx % 8;
    return col * 8 + row;
}

int transform_transpose_anti_py(int move_idx) {
    int row = move_idx / 8;
    int col = move_idx % 8;
    return (7 - col) * 8 + (7 - row);
}

PYBIND11_MODULE(reversi_bitboard_cpp, m) {
    m.doc() = "pybind11 plugin for ReversiBitboard";

    py::class_<ReversiBitboard>(m, "ReversiBitboard")
        .def(py::init<>())
        .def("reset", &ReversiBitboard::reset)
        .def_readwrite("black_board", &ReversiBitboard::black_board)
        .def_readwrite("white_board", &ReversiBitboard::white_board)
        .def_readwrite("current_player", &ReversiBitboard::current_player)
        .def_readwrite("passed_last_turn", &ReversiBitboard::passed_last_turn)
        .def_readwrite("history", &ReversiBitboard::history)
        .def("get_legal_moves_bitboard", &ReversiBitboard::get_legal_moves_bitboard)
        .def("apply_move", &ReversiBitboard::apply_move)
        .def("is_game_over", &ReversiBitboard::is_game_over)
        .def("get_winner", &ReversiBitboard::get_winner)
        .def("count_set_bits", &ReversiBitboard::count_set_bits)
        .def("get_legal_moves", &ReversiBitboard::get_legal_moves)
        .def("board_to_numpy", [](const ReversiBitboard& self) {
            std::vector<int8_t> board_vec = self.board_to_numpy();
            return py::array_t<int8_t>(board_vec.size(), board_vec.data());
        })
        .def("board_to_input_planes", [](const ReversiBitboard& self, int current_player) {
            std::vector<int8_t> board_1d = self.board_to_numpy();
            py::array_t<float> player_plane = py::array_t<float>({8, 8});
            py::array_t<float> opponent_plane = py::array_t<float>({8, 8});

            auto buf_player = player_plane.request();
            auto ptr_player = static_cast<float*>(buf_player.ptr);
            auto buf_opponent = opponent_plane.request();
            auto ptr_opponent = static_cast<float*>(buf_opponent.ptr);
            for (int r = 0; r < 8; ++r) {
                for (int c = 0; c < 8; ++c) {
                    ptr_player[r * 8 + c] = 0.0f;
                    ptr_opponent[r * 8 + c] = 0.0f;
                }
            }

            int opponent = 3 - current_player;

            for (int i = 0; i < ReversiBitboard::BOARD_SIZE; ++i) {
                int r = i / ReversiBitboard::BOARD_LENGTH;
                int c = i % ReversiBitboard::BOARD_LENGTH;
                if ((self.black_board & (1ULL << i)) && (self.white_board & (1ULL << i))) {
                    continue;
                }
                if (board_1d[i] == current_player) {
                    ptr_player[r * ReversiBitboard::BOARD_LENGTH + c] = 1.0f;
                } else if (board_1d[i] == opponent) {
                    ptr_opponent[r * ReversiBitboard::BOARD_LENGTH + c] = 1.0f;
                }
            }

            py::array_t<float> result = py::array_t<float>({8, 8, 2});
            auto buf_result = result.request();
            auto ptr_result = static_cast<float*>(buf_result.ptr);

            for (int r = 0; r < 8; ++r) {
                for (int c = 0; c < 8; ++c) {
                    ptr_result[(r * 8 + c) * 2 + 0] = ptr_player[r * 8 + c];
                    ptr_result[(r * 8 + c) * 2 + 1] = ptr_opponent[r * 8 + c];
                }
            }
            return result;
        }, py::arg("current_player"))
        .def("flip_horizontal", &ReversiBitboard::flip_horizontal)
        .def("flip_vertical", &ReversiBitboard::flip_vertical)
        .def("transpose_main", &ReversiBitboard::transpose_main)
        .def("transpose_anti", &ReversiBitboard::transpose_anti)
        ;

    m.def("transform_policy_horizontal", &transform_horizontal_py, "Transforms a policy index for horizontal flip");
    m.def("transform_policy_vertical", &transform_vertical_py, "Transforms a policy index for vertical flip");
    m.def("transform_policy_transpose_main", &transform_transpose_main_py, "Transforms a policy index for main diagonal transpose");
    m.def("transform_policy_transpose_anti", &transform_transpose_anti_py, "Transforms a policy index for anti-diagonal transpose");
}