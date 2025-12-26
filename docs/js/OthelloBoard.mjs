export class OthelloBoard {
    constructor() {
        this.blackBoard = 0x0000001008000000n;
        this.whiteBoard = 0x0000000810000000n;
        this.currentPlayer = 1;
        this.passedLastTurn = false;
        this.history = [];

        this.BOARD_SIZE = 64;
        this.BOARD_LENGTH = 8;
        this.ALL_MASK = 0xFFFFFFFFFFFFFFFFn;
        this.L_MASK = 0x0101010101010101n;
        this.R_MASK = 0x8080808080808080n;
    }

    reset() {
        this.blackBoard = 0x0000001008000000n;
        this.whiteBoard = 0x0000000810000000n;
        this.currentPlayer = 1;
        this.passedLastTurn = false;
        this.history = [];
    }

    clone() {
        const newBoard = new OthelloBoard();
        newBoard.blackBoard = this.blackBoard;
        newBoard.whiteBoard = this.whiteBoard;
        newBoard.currentPlayer = this.currentPlayer;
        newBoard.passedLastTurn = this.passedLastTurn;
        newBoard.history = [...this.history];
        return newBoard;
    }

    countSetBits(n) {
        let count = 0;
        while (n > 0n) {
            n &= (n - 1n);
            count++;
        }
        return count;
    }

    getRowCol(index) {
        return { row: Math.floor(index / this.BOARD_LENGTH), col: index % this.BOARD_LENGTH };
    }

    getIndex(row, col) {
        return row * this.BOARD_LENGTH + col;
    }

    _calculateFlips(moveBit, playerBoard, enemyBoard) {
        let moveMask = 1n << BigInt(moveBit);
        let totalFlipMask = 0n;
        const directions = [-9, -8, -7, -1, 1, 7, 8, 9];

        for (let shift of directions) {
            let line = 0n;
            let current = moveMask;
            for (let i = 0; i < this.BOARD_LENGTH; ++i) {
                if ((shift === 1 || shift === -7 || shift === 9) && (current & this.R_MASK)) {
                    break;
                }
                if ((shift === -1 || shift === 7 || shift === -9) && (current & this.L_MASK)) {
                    break;
                }
                if (shift > 0) {
                    current <<= BigInt(shift);
                } else {
                    current >>= BigInt(-shift);
                }
                if (current === 0n) {
                    line = 0n;
                    break;
                }
                if ((current & enemyBoard)) {
                    line |= current;
                } else if ((current & playerBoard)) {
                    totalFlipMask |= line;
                    break;
                } else {
                    line = 0n;
                    break;
                }
            }
        }
        return totalFlipMask;
    }

    getLegalMovesBitboard() {
        const playerBoard = (this.currentPlayer === 1) ? this.blackBoard : this.whiteBoard;
        const enemyBoard = (this.currentPlayer === 1) ? this.whiteBoard : this.blackBoard;
        const emptySquares = ~(playerBoard | enemyBoard) & this.ALL_MASK;
        let legalMoves = 0n;

        for (let i = 0; i < this.BOARD_SIZE; ++i) {
            if (((1n << BigInt(i)) & emptySquares)) {
                if (this._calculateFlips(i, playerBoard, enemyBoard) !== 0n) {
                    legalMoves |= (1n << BigInt(i));
                }
            }
        }
        return legalMoves;
    }

    applyMove(moveBit) {
        this.history.push(moveBit);
        if (moveBit === -1) {
            this.passedLastTurn = true;
            this.currentPlayer = 3 - this.currentPlayer;
            return;
        }

        let playerBoardMask = (this.currentPlayer === 1) ? this.blackBoard : this.whiteBoard;
        let enemyBoardMask = (this.currentPlayer === 1) ? this.whiteBoard : this.blackBoard;

        const flipMask = this._calculateFlips(moveBit, playerBoardMask, enemyBoardMask);
        const moveMask = 1n << BigInt(moveBit);

        playerBoardMask |= (moveMask | flipMask);
        enemyBoardMask ^= flipMask;

        if (this.currentPlayer === 1) {
            this.blackBoard = playerBoardMask;
            this.whiteBoard = enemyBoardMask;
        } else {
            this.whiteBoard = playerBoardMask;
            this.blackBoard = enemyBoardMask;
        }

        this.passedLastTurn = false;
        this.currentPlayer = 3 - this.currentPlayer;
    }

    getLegalMoves() {
        const bitboard = this.getLegalMovesBitboard();
        const moves = [];
        for (let i = 0; i < this.BOARD_SIZE; ++i) {
            if (((bitboard >> BigInt(i)) & 1n)) {
                moves.push(i);
            }
        }
        return moves;
    }

    isGameOver() {
        if ((this.blackBoard | this.whiteBoard) === this.ALL_MASK) {
            return true;
        }
        if (this.blackBoard === 0n || this.whiteBoard === 0n) {
            return true;
        }
        if (this.getLegalMovesBitboard() !== 0n) {
            return false;
        }
        const tempBoard = new OthelloBoard();
        tempBoard.blackBoard = this.blackBoard;
        tempBoard.whiteBoard = this.whiteBoard;
        tempBoard.currentPlayer = 3 - this.currentPlayer;
        return tempBoard.getLegalMovesBitboard() === 0n;
    }

    getWinner() {
        const blackCount = this.countSetBits(this.blackBoard);
        const whiteCount = this.countSetBits(this.whiteBoard);
        if (blackCount > whiteCount) return 1;
        else if (whiteCount > blackCount) return 2;
        else return 0;
    }

    boardToInputPlanes() {
        const playerPlane = new Array(this.BOARD_SIZE).fill(0);
        const opponentPlane = new Array(this.BOARD_SIZE).fill(0);

        const playerBoard = (this.currentPlayer === 1) ? this.blackBoard : this.whiteBoard;
        const enemyBoard = (this.currentPlayer === 1) ? this.whiteBoard : this.blackBoard;

        for (let i = 0; i < this.BOARD_SIZE; ++i) {
            const mask = 1n << BigInt(i);
            if ((playerBoard & mask)) {
                playerPlane[i] = 1;
            } else if ((enemyBoard & mask)) {
                opponentPlane[i] = 1;
            }
        }

        const inputPlanes = [];
        for (let r = 0; r < this.BOARD_LENGTH; ++r) {
            const row = [];
            for (let c = 0; c < this.BOARD_LENGTH; ++c) {
                const index = r * this.BOARD_LENGTH + c;
                row.push([playerPlane[index], opponentPlane[index]]);
            }
            inputPlanes.push(row);
        }
        return inputPlanes;
    }
}
