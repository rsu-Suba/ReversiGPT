import { OthelloBoard } from "./js/OthelloBoard.mjs";
import { MCTS } from "./js/MCTS.mjs";

const boardElement = document.getElementById("board");
const currentPlayerSpan = document.getElementById("current-player");
const blackScoreSpan = document.getElementById("black-score");
const whiteScoreSpan = document.getElementById("white-score");
const gameMessage = document.getElementById("game-message");
const tempMessage = document.getElementById("temp-message");
const resetButton = document.getElementById("reset-button");

let gameBoard;
let aiModel;
let humanPlayer;

const MCTS_SIMS = 20;
const MCTS_PUCT = 3.0;

async function initGame() {
   await tf.ready();
   gameBoard = new OthelloBoard();
   await loadModel();

   console.log(`AI Configuration: Sims=${MCTS_SIMS}, PUCT=${MCTS_PUCT}`);

   humanPlayer = Math.random() < 0.5 ? 1 : 2;
   gameBoard.currentPlayer = 1;

   renderBoard();
   updateGameInfo();

   if (gameBoard.currentPlayer !== humanPlayer) {
      setTimeout(makeAIMove, 500);
   }
}

async function loadModel() {
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
            cell.classList.add("valid-move");
            cell.addEventListener("click", () => handleMove(i));
         }
      }

      boardElement.appendChild(cell);
   }
}

function updateGameInfo() {
   const currentPlayerName = gameBoard.currentPlayer === 1 ? "Black" : "White";
   const humanPlayerName = humanPlayer === 1 ? "Black (You)" : "White (You)";
   const aiPlayerName = humanPlayer === 1 ? "White (AI)" : "Black (AI)";

   currentPlayerSpan.textContent = `${ 
      gameBoard.currentPlayer === humanPlayer ? "Player" : "AI"
   } (${currentPlayerName})`;
   blackScoreSpan.textContent = gameBoard.countSetBits(gameBoard.blackBoard);
   whiteScoreSpan.textContent = gameBoard.countSetBits(gameBoard.whiteBoard);

   gameMessage.textContent = "";

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
      gameMessage.textContent = message;
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
   tempMessage.classList.add('show');
   setTimeout(() => {
      tempMessage.classList.remove('show');
   }, 1500);
}

async function handleMove(move) {
   if (gameBoard.currentPlayer === humanPlayer) {
      gameBoard.applyMove(move);
      renderBoard();
      updateGameInfo();
      if (!gameBoard.isGameOver() && gameBoard.currentPlayer !== humanPlayer) {
         setTimeout(makeAIMove, 0);
      }
   }
}

async function makeAIMove() {
   if (gameBoard.isGameOver()) return;
   console.log("AI is thinking...");
   
   // Allow UI to update before heavy calculation
   await new Promise(r => setTimeout(r, 50));

   const mcts = new MCTS(aiModel, MCTS_PUCT);
   // Run simulations
   const root = await mcts.search(gameBoard, MCTS_SIMS);

   const children = Object.values(root.children);
   if (children.length === 0) {
       console.warn("AI has no moves.");
       return;
   }

   // Sort by visits (descending)
   children.sort((a, b) => b.visits - a.visits);
   
   const bestChild = children[0];
   const bestMove = bestChild.move;

   // Log Top 5
   let logMsg = "--- AI Search Results ---\n";
   logMsg += `Best Move: ${indexToCoord(bestMove)} (Visits: ${bestChild.visits}, Q: ${(-bestChild.qValue).toFixed(4)})\n`;
   
   children.slice(0, 5).forEach((child, i) => {
       logMsg += `${i+1}. ${indexToCoord(child.move)}: Visits=${child.visits}, Q=${(-child.qValue).toFixed(4)}\n`;
   });
   logMsg += "-------------------------";
   console.log(logMsg);

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

resetButton.addEventListener("click", () => {
   gameBoard.reset();
   initGame();
});

initGame();