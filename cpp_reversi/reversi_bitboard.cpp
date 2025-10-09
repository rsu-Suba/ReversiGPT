#include "reversi_bitboard.h"
#include <iostream>

ReversiBitboard::ReversiBitboard() {
    reset();
}

void ReversiBitboard::reset() {
    black_board = BLACK_INIT_BOARD;
    white_board = WHITE_INIT_BOARD;
    current_player = 1;
    passed_last_turn = false;
}

uint64_t ReversiBitboard::_calculate_flips(int move_bit, uint64_t player_board, uint64_t enemy_board) const {
    uint64_t move_mask = 1ULL << move_bit;
    uint64_t total_flip_mask = 0ULL;
    const int directions[] = {-9, -8, -7, -1, 1, 7, 8, 9};

    for (int shift : directions) {
        uint64_t line = 0ULL;
        uint64_t current = move_mask;
        for (int i = 0; i < BOARD_LENGTH; ++i) {
            if ((shift == 1 || shift == -7 || shift == 9) && (current & R_MASK)) {
                break;
            }
            if ((shift == -1 || shift == 7 || shift == -9) && (current & L_MASK)) {
                break;
            }
            if (shift > 0) {
                current <<= shift;
            } else {
                current >>= -shift;
            }
            if (!(current & ALL_MASK)) {
                line = 0ULL;
                break;
            }

            if ((current & enemy_board)) {
                line |= current;
            } else if ((current & player_board)) {
                total_flip_mask |= line;
                break;
            } else {
                line = 0ULL;
                break;
            }
        }
    }
    return total_flip_mask;
}

uint64_t ReversiBitboard::get_legal_moves_bitboard() const {
    uint64_t player_board = (current_player == 1) ? black_board : white_board;
    uint64_t enemy_board = (current_player == 1) ? white_board : black_board;
    uint64_t empty_squares = ~(player_board | enemy_board) & ALL_MASK;
    uint64_t legal_moves = 0ULL;

    for (int i = 0; i < BOARD_SIZE; ++i) {
        if (((1ULL << i) & empty_squares)) {
            if (_calculate_flips(i, player_board, enemy_board) != 0ULL) {
                legal_moves |= (1ULL << i);
            }
        }
    }
    return legal_moves;
}

void ReversiBitboard::apply_move(int move_bit) {
    history.push_back(move_bit);
    if (move_bit == -1) {
        passed_last_turn = true;
        current_player = 3 - current_player;
        return;
    }

    uint64_t player_board_mask = (current_player == 1) ? black_board : white_board;
    uint64_t enemy_board_mask = (current_player == 1) ? white_board : black_board;

    uint64_t flip_mask = _calculate_flips(move_bit, player_board_mask, enemy_board_mask);
    uint64_t move_mask = 1ULL << move_bit;

    player_board_mask |= (move_mask | flip_mask);
    enemy_board_mask ^= flip_mask;

    if (current_player == 1) {
        black_board = player_board_mask;
        white_board = enemy_board_mask;
    } else {
        white_board = player_board_mask;
        black_board = enemy_board_mask;
    }

    passed_last_turn = false;
    current_player = 3 - current_player;
}

bool ReversiBitboard::is_game_over() const {
    if ((black_board | white_board) == ALL_MASK) {
        return true;
    }
    if (black_board == 0ULL || white_board == 0ULL) {
        return true;
    }
    if (get_legal_moves_bitboard() != 0ULL) {
        return false;
    }
    ReversiBitboard temp_board = *this;
    temp_board.current_player = 3 - temp_board.current_player;
    return temp_board.get_legal_moves_bitboard() == 0ULL;
}

int ReversiBitboard::get_winner() const {
    int black_count = count_set_bits(black_board);
    int white_count = count_set_bits(white_board);
    if (black_count > white_count) return 1;
    else if (white_count > black_count) return 2;
    else return 0;
}

int ReversiBitboard::count_set_bits(uint64_t n) const {
    int count = 0;
    while (n > 0ULL) {
        n &= (n - 1ULL);
        count++;
    }
    return count;
}

std::vector<int> ReversiBitboard::get_legal_moves() const {
    uint64_t bitboard = get_legal_moves_bitboard();
    std::vector<int> moves;
    for (int i = 0; i < BOARD_SIZE; ++i) {
        if ((bitboard >> i) & 1ULL) {
            moves.push_back(i);
        }
    }
    return moves;
}

std::vector<int8_t> ReversiBitboard::board_to_numpy() const {
    std::vector<int8_t> board_1d(BOARD_SIZE);
    for (int i = 0; i < BOARD_SIZE; ++i) {
        uint64_t mask = 1ULL << i;
        if ((black_board & mask)) {
            board_1d[i] = 1;
        } else if ((white_board & mask)) {
            board_1d[i] = 2;
        } else {
            board_1d[i] = 0;
        }
    }
    return board_1d;
}

int get_row(int bit_index) { return bit_index / ReversiBitboard::BOARD_LENGTH; }
int get_col(int bit_index) { return bit_index % ReversiBitboard::BOARD_LENGTH; }
int get_bit_index(int row, int col) { return row * ReversiBitboard::BOARD_LENGTH + col; }

int transform_horizontal(int bit_index) {
    int row = get_row(bit_index);
    int col = get_col(bit_index);
    return get_bit_index(row, 7 - col);
}

int transform_vertical(int bit_index) {
    int row = get_row(bit_index);
    int col = get_col(bit_index);
    return get_bit_index(7 - row, col);
}

int transform_transpose_main(int bit_index) {
    int row = get_row(bit_index);
    int col = get_col(bit_index);
    return get_bit_index(col, row);
}

int transform_transpose_anti(int bit_index) {
    int row = get_row(bit_index);
    int col = get_col(bit_index);
    return get_bit_index(7 - col, 7 - row);
}

uint64_t apply_board_transform(uint64_t board, int (*transform_func)(int)) {
    uint64_t transformed_board = 0ULL;
    for (int i = 0; i < ReversiBitboard::BOARD_SIZE; ++i) {
        if ((board >> i) & 1ULL) {
            transformed_board |= (1ULL << transform_func(i));
        }
    }
    return transformed_board;
}

ReversiBitboard ReversiBitboard::flip_horizontal() const {
    ReversiBitboard new_board = *this;
    new_board.black_board = apply_board_transform(black_board, transform_horizontal);
    new_board.white_board = apply_board_transform(white_board, transform_horizontal);
    return new_board;
}

ReversiBitboard ReversiBitboard::flip_vertical() const {
    ReversiBitboard new_board = *this;
    new_board.black_board = apply_board_transform(black_board, transform_vertical);
    new_board.white_board = apply_board_transform(white_board, transform_vertical);
    return new_board;
}

ReversiBitboard ReversiBitboard::transpose_main() const {
    ReversiBitboard new_board = *this;
    new_board.black_board = apply_board_transform(black_board, transform_transpose_main);
    new_board.white_board = apply_board_transform(white_board, transform_transpose_main);
    return new_board;
}

ReversiBitboard ReversiBitboard::transpose_anti() const {
    ReversiBitboard new_board = *this;
    new_board.black_board = apply_board_transform(black_board, transform_transpose_anti);
    new_board.white_board = apply_board_transform(white_board, transform_transpose_anti);
    return new_board;
}