#ifndef REVERSI_BITBOARD_H
#define REVERSI_BITBOARD_H

#include <cstdint>
#include <vector>
#include <numeric>

class ReversiBitboard {
public:
    uint64_t black_board;
    uint64_t white_board;
    int current_player;
    bool passed_last_turn;
    std::vector<int> history;

    static const int BOARD_SIZE = 64;
    static const int BOARD_LENGTH = 8;
    static const uint64_t ALL_MASK = ~0ULL;
    static const uint64_t U_MASK = 0xffULL;
    static const uint64_t D_MASK = 0xff0000000000ULL;
    static const uint64_t L_MASK = 0x0101010101010101ULL;
    static const uint64_t R_MASK = 0x8080808080808080ULL;
    static const uint64_t BLACK_INIT_BOARD = 0x0000001008000000ULL;
    static const uint64_t WHITE_INIT_BOARD = 0x0000000810000000ULL;

    ReversiBitboard();
    void reset();

    uint64_t get_legal_moves_bitboard() const;
    void apply_move(int move_bit);
    bool is_game_over() const;
    int get_winner() const;
    int count_set_bits(uint64_t n) const;

    std::vector<int> get_legal_moves() const;
    std::vector<int> get_flipped_indices(int move_bit) const;
    std::vector<int8_t> board_to_numpy() const;

    ReversiBitboard flip_horizontal() const;
    ReversiBitboard flip_vertical() const;
    ReversiBitboard transpose_main() const;
    ReversiBitboard transpose_anti() const;

private:
    uint64_t _calculate_flips(int move_bit, uint64_t player_board, uint64_t enemy_board) const;
};

#endif