import { OthelloBoard } from "./js/OthelloBoard.mjs";
import { MCTS } from "./js/MCTS.mjs";

const boardElement = document.getElementById("board");
const currentPlayerSpan = document.getElementById("current-player");
const blackScoreSpan = document.getElementById("black-score");
const whiteScoreSpan = document.getElementById("white-score");
const gameMessage = document.getElementById("game-message");
const tempMessage = document.getElementById("temp-message");
const resetButton = document.getElementById("reset-button");
const reviewButton = document.getElementById("review-button");
const mainOverlay = document.getElementById("main-overlay");

const settingsToggle = document.getElementById("settings-button");
const settingsPanel = document.getElementById("settings-panel");
const optHeatmap = document.getElementById("opt-heatmap");
const optLines = document.getElementById("opt-lines");
const optLegal = document.getElementById("opt-legal");

const debugView = document.getElementById("debug-view");
const debugStartBtn = document.getElementById("debug-start");
const debugPrevBtn = document.getElementById("debug-prev");
const debugNextBtn = document.getElementById("debug-next");
const debugEndBtn = document.getElementById("debug-end");
const debugCloseBtn = document.getElementById("debug-close");
const debugCounter = document.getElementById("debug-counter");
const debugBoardLeft = document.getElementById("debug-board-left");
const debugBoardRight = document.getElementById("debug-board-right");
const debugOverlay = document.getElementById("debug-overlay");

let gameBoard;
let aiModel;
let humanPlayer;
let replayIndex = 0;

let settings = {
    heatmap: true,
    lines: true,
    legalMoves: true,
};

const MCTS_SIMS = 25;
const MCTS_PUCT = 1.3;

async function initGame() {
    await tf.ready();
    loadSettings();
    initSettingsListeners();

    gameBoard = new OthelloBoard();
    await loadModel();

    console.log(`AI Configuration: Sims=${MCTS_SIMS}, PUCT=${MCTS_PUCT}`);

    humanPlayer = Math.random() < 0.5 ? 1 : 2;
    gameBoard.currentPlayer = 1;

    renderBoard();
    updateGameInfo();

    reviewButton.style.display = "none";
    debugView.classList.add("hidden");

    if (gameBoard.currentPlayer !== humanPlayer) {
        setTimeout(makeAIMove, 0);
    }
}

async function loadModel() {
    if (aiModel) return;
    console.log("Loading AI model...");
    aiModel = await tf.loadGraphModel("./tfjs_model/model.json");
    console.log("AI model loaded.");
}

function renderBoard() {
    boardElement.innerHTML = "";
    const blackPieces = gameBoard.blackBoard;
    const whitePieces = gameBoard.whiteBoard;
    const legalMoves = gameBoard.getLegalMoves();

    for (let i = 0; i < gameBoard.BOARD_SIZE; i++) {
        const cell = document.createElement("div");
        cell.classList.add("cell");
        cell.dataset.index = i;

        if ((blackPieces >> BigInt(i)) & 1n) {
            const piece = document.createElement("div");
            piece.classList.add("piece", "black");
            cell.appendChild(piece);
        } else if ((whitePieces >> BigInt(i)) & 1n) {
            const piece = document.createElement("div");
            piece.classList.add("piece", "white");
            cell.appendChild(piece);
        }

        if (legalMoves.includes(i)) {
            if (gameBoard.currentPlayer === humanPlayer) {
                if (settings.legalMoves) {
                    cell.classList.add("valid-move");
                }
                cell.addEventListener("click", () => handleMove(i));
            }
        }

        boardElement.appendChild(cell);
    }
}

function updateGameInfo() {
    const currentPlayerName = gameBoard.currentPlayer === 1 ? "Black" : "White";
    const humanPlayerName = humanPlayer === 1 ? "Black (You)" : "White (You)";

    currentPlayerSpan.textContent = `${
        gameBoard.currentPlayer === humanPlayer ? "Player" : "AI"
    } (${currentPlayerName})`;
    blackScoreSpan.textContent = gameBoard.countSetBits(gameBoard.blackBoard);
    whiteScoreSpan.textContent = gameBoard.countSetBits(gameBoard.whiteBoard);

    gameMessage.textContent = "";
    gameMessage.style.padding = 0;
    gameMessage.style.marginTop = 0;

    if (gameBoard.isGameOver()) {
        const winner = gameBoard.getWinner();
        let message = "";
        if (winner === 1) {
            message = `${humanPlayer === 1 ? "You" : "AI"} wins`;
        } else if (winner === 2) {
            message = `${humanPlayer === 1 ? "AI" : "You"} wins`;
        } else {
            message = "Draw";
        }
        gameMessage.style.padding = "10px";
        gameMessage.style.marginTop = "15px";
        gameMessage.textContent = message;
        reviewButton.style.display = "inline-block";
        clearVisualization();
    } else if (gameBoard.getLegalMoves().length === 0) {
        const tempBoard = new OthelloBoard();
        tempBoard.blackBoard = gameBoard.blackBoard;
        tempBoard.whiteBoard = gameBoard.whiteBoard;
        tempBoard.currentPlayer = 3 - gameBoard.currentPlayer;

        if (tempBoard.getLegalMoves().length === 0) {
            gameBoard.isGameOver();
            updateGameInfo();
        } else {
            const passedPlayerName = gameBoard.currentPlayer === humanPlayer ? "Player" : "AI";
            showTempMessage(`${passedPlayerName} passed!`);
            gameBoard.applyMove(-1);
            renderBoard();
            updateGameInfo();
            if (gameBoard.currentPlayer !== humanPlayer) {
                setTimeout(makeAIMove, 500);
            }
        }
    }
}

function showTempMessage(message) {
    tempMessage.textContent = message;
    tempMessage.classList.add("show");
    setTimeout(() => {
        tempMessage.classList.remove("show");
    }, 1500);
}

async function handleMove(move) {
    if (gameBoard.currentPlayer === humanPlayer) {
        clearVisualization();
        gameBoard.applyMove(move);
        renderBoard();
        updateGameInfo();
        if (!gameBoard.isGameOver() && gameBoard.currentPlayer !== humanPlayer) {
            setTimeout(makeAIMove, 0);
        }
    }
}

function clearVisualization() {
    mainOverlay.innerHTML = "";
    const cells = document.querySelectorAll("#board .cell");
    cells.forEach((cell) => {
        cell.style.backgroundColor = "";
    });
}

function renderThinking(root) {
    const children = Object.values(root.children);
    if (children.length === 0) return;
    let maxVisits = 0;
    children.forEach((child) => {
        if (child.visits > maxVisits) maxVisits = child.visits;
    });
    const cells = document.querySelectorAll("#board .cell");
    if (settings.heatmap) {
        children.forEach((child) => {
            if (child.move === -1) return;

            const cell = cells[child.move];
            if (cell) {
                const opacity = maxVisits > 0 ? (child.visits / maxVisits) * 0.6 : 0;
                cell.style.backgroundColor = `rgba(255, 50, 50, ${opacity})`;
            }
        });
    }

    if (settings.lines) {
        children.sort((a, b) => b.visits - a.visits);
        const bestChild = children[0];
        if (bestChild && bestChild.move !== -1 && bestChild.visits > 1) {
            const lines = calculateFlipLines(gameBoard, bestChild.move);
            const firstCell = document.querySelector("#board .cell");
            const cellSize = firstCell ? firstCell.getBoundingClientRect().width : 55;
            drawLines(lines, mainOverlay, cellSize);
        }
    }
}

async function makeAIMove() {
    if (gameBoard.isGameOver()) return;
    clearVisualization();
    console.log("AI is thinking...");
    await new Promise((r) => setTimeout(r, 1));
    const mcts = new MCTS(aiModel, MCTS_PUCT);
    const root = await mcts.search(gameBoard, MCTS_SIMS, (r) => renderThinking(r));
    const children = Object.values(root.children);
    if (children.length === 0) {
        console.warn("AI has no moves.");
        return;
    }

    children.sort((a, b) => b.visits - a.visits);
    const bestChild = children[0];
    const bestMove = bestChild.move;
    let logMsg =
        "--- AI Search Results ---\
";
    logMsg += `Best Move: ${indexToCoord(bestMove)} (Visits: ${bestChild.visits}, Q: ${(-bestChild.qValue).toFixed(4)})
`;
    children.slice(0, 5).forEach((child, i) => {
        logMsg += `${i + 1}. ${indexToCoord(child.move)}: Visits=${child.visits}, Q=${(-child.qValue).toFixed(4)}
`;
    });
    logMsg += "-------------------------";
    console.log(logMsg);

    await new Promise((r) => setTimeout(r, 5));
    clearVisualization();

    if (bestMove !== -1) {
        gameBoard.applyMove(bestMove);
    } else {
        console.log("AI Passes.");
        gameBoard.applyMove(-1);
    }

    renderBoard();
    updateGameInfo();
}

function indexToCoord(index) {
    if (index === -1) return "PASS";
    const r = Math.floor(index / 8);
    const c = index % 8;
    return String.fromCharCode(65 + c) + (r + 1);
}

reviewButton.addEventListener("click", () => {
    startDebugSession();
});

debugCloseBtn.addEventListener("click", () => {
    debugView.classList.add("hidden");
});

debugStartBtn.addEventListener("click", () => {
    replayIndex = 0;
    renderDebugStep(replayIndex);
});

function setupLongPress(btn, action) {
    let timeout;
    let interval;

    const start = (e) => {
        if (e.type === "touchstart") e.preventDefault();
        action();
        timeout = setTimeout(() => {
            interval = setInterval(action, 200);
        }, 500);
    };

    const stop = () => {
        clearTimeout(timeout);
        clearInterval(interval);
    };

    btn.addEventListener("mousedown", start);
    btn.addEventListener("touchstart", start);
    btn.addEventListener("mouseup", stop);
    btn.addEventListener("mouseleave", stop);
    btn.addEventListener("touchend", stop);
}

setupLongPress(debugPrevBtn, () => {
    if (replayIndex > 0) {
        replayIndex--;
        renderDebugStep(replayIndex);
    }
});

setupLongPress(debugNextBtn, () => {
    if (replayIndex < gameBoard.history.length - 1) {
        replayIndex++;
        renderDebugStep(replayIndex);
    }
});

debugEndBtn.addEventListener("click", () => {
    replayIndex = gameBoard.history.length - 1;
    if (replayIndex < 0) replayIndex = 0;
    renderDebugStep(replayIndex);
});

function startDebugSession() {
    debugView.classList.remove("hidden");
    replayIndex = 0;
    if (gameBoard.history.length > 0) {
        renderDebugStep(0);
    }
    debugView.scrollIntoView({ behavior: "smooth" });
}

function renderSmallBoard(boardInstance, elementId, highlightMove) {
    const el = document.getElementById(elementId);
    el.innerHTML = "";
    const blackPieces = boardInstance.blackBoard;
    const whitePieces = boardInstance.whiteBoard;

    for (let i = 0; i < 64; i++) {
        const cell = document.createElement("div");
        cell.classList.add("cell");
        if (i === highlightMove) {
            cell.classList.add("just-placed");
        }

        if ((blackPieces >> BigInt(i)) & 1n) {
            const piece = document.createElement("div");
            piece.classList.add("piece", "black");
            cell.appendChild(piece);
        } else if ((whitePieces >> BigInt(i)) & 1n) {
            const piece = document.createElement("div");
            piece.classList.add("piece", "white");
            cell.appendChild(piece);
        }
        el.appendChild(cell);
    }
}

function renderDebugStep(index) {
    if (index < 0 || index >= gameBoard.history.length) return;
    debugCounter.textContent = `Move: ${index + 1} / ${gameBoard.history.length}`;
    const tempBoard = new OthelloBoard();
    for (let i = 0; i < index; i++) {
        tempBoard.applyMove(gameBoard.history[i]);
    }
    const currentMove = gameBoard.history[index];
    renderSmallBoard(tempBoard, "debug-board-left", currentMove);
    let lines = [];
    if (currentMove !== -1) {
        lines = calculateFlipLines(tempBoard, currentMove);
    }

    tempBoard.applyMove(currentMove);
    renderSmallBoard(tempBoard, "debug-board-right", -1);
    drawDebugLines(lines);
}

function calculateFlipLines(board, moveBit) {
    if (moveBit === -1) return [];

    const lines = [];
    const playerBoard = board.currentPlayer === 1 ? board.blackBoard : board.whiteBoard;
    const enemyBoard = board.currentPlayer === 1 ? board.whiteBoard : board.blackBoard;

    const directions = [
        { shift: -9, dr: -1, dc: -1 },
        { shift: -8, dr: -1, dc: 0 },
        { shift: -7, dr: -1, dc: 1 },
        { shift: -1, dr: 0, dc: -1 },
        { shift: 1, dr: 0, dc: 1 },
        { shift: 7, dr: 1, dc: -1 },
        { shift: 8, dr: 1, dc: 0 },
        { shift: 9, dr: 1, dc: 1 },
    ];

    const moveMask = 1n << BigInt(moveBit);
    const { row: startR, col: startC } = board.getRowCol(moveBit);

    for (const dir of directions) {
        let current = moveMask;
        let potentiallyFlipped = 0n;
        let r = startR;
        let c = startC;

        for (let i = 0; i < 8; i++) {
            r += dir.dr;
            c += dir.dc;
            if (r < 0 || r >= 8 || c < 0 || c >= 8) break;

            const nextBit = BigInt(r * 8 + c);
            const nextMask = 1n << nextBit;

            if (nextMask & enemyBoard) {
                potentiallyFlipped |= nextMask;
            } else if (nextMask & playerBoard) {
                if (potentiallyFlipped !== 0n) {
                    lines.push({
                        x1: startC,
                        y1: startR,
                        x2: c,
                        y2: r,
                    });
                }
                break;
            } else {
                break;
            }
        }
    }
    return lines;
}

function drawLines(lines, targetSvg, cellSize) {
    targetSvg.innerHTML = "";
    const offset = cellSize / 2;

    lines.forEach((line) => {
        const x1 = line.x1 * cellSize + offset;
        const y1 = line.y1 * cellSize + offset;
        const x2 = line.x2 * cellSize + offset;
        const y2 = line.y2 * cellSize + offset;

        const svgLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
        svgLine.setAttribute("x1", x1);
        svgLine.setAttribute("y1", y1);
        svgLine.setAttribute("x2", x2);
        svgLine.setAttribute("y2", y2);
        svgLine.classList.add("debug-line");

        targetSvg.appendChild(svgLine);
    });
}

function drawDebugLines(lines) {
    drawLines(lines, debugOverlay, 40);
}

resetButton.addEventListener("click", () => {
    gameBoard.reset();
    initGame();
});

function loadSettings() {
    try {
        const saved = localStorage.getItem("reversiSettings");
        if (saved) {
            settings = JSON.parse(saved);
        }
    } catch (e) {
        console.error("Failed to load settings", e);
    }
    optHeatmap.checked = settings.heatmap;
    optLines.checked = settings.lines;
    optLegal.checked = settings.legalMoves;
}

function saveSettings() {
    settings.heatmap = optHeatmap.checked;
    settings.lines = optLines.checked;
    settings.legalMoves = optLegal.checked;
    localStorage.setItem("reversiSettings", JSON.stringify(settings));
    renderBoard();
    if (!settings.heatmap) {
        const cells = document.querySelectorAll("#board .cell");
        cells.forEach((cell) => (cell.style.backgroundColor = ""));
    }
    if (!settings.lines) {
        mainOverlay.innerHTML = "";
    }
}

function initSettingsListeners() {
    settingsToggle.addEventListener("click", () => {
        settingsPanel.classList.toggle("open");
    });

    optHeatmap.addEventListener("change", saveSettings);
    optLines.addEventListener("change", saveSettings);
    optLegal.addEventListener("change", saveSettings);
}

initGame();
