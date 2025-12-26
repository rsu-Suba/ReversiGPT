import { OthelloBoard } from "./js/OthelloBoard.mjs";

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

async function initGame() {
   await tf.ready();
   gameBoard = new OthelloBoard();
   await loadModel();

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
   setTimeout(async () => {
      const inputTensor = tf.tensor4d([gameBoard.boardToInputPlanes()], [1, 8, 8, 2], "float32");
      const predictions = aiModel.predict(inputTensor);
      
      // Debug: Check output shapes
      if (Array.isArray(predictions)) {
         // console.log("Predictions shapes:", predictions.map(p => p.shape));
      } else {
         // console.log("Prediction shape:", predictions.shape);
      }

      // predictions[0] should be Policy (64), predictions[1] is Value (1)
      // Check which one has size 64
      let policyOutput;
      if (Array.isArray(predictions)) {
          if (predictions[0].size === 64) {
              policyOutput = predictions[0].dataSync();
          } else {
              policyOutput = predictions[1].dataSync();
          }
      } else {
          policyOutput = predictions.dataSync();
      }

      const legalMoves = gameBoard.getLegalMoves();
      let bestMove = -1;
      let maxPolicy = -1;

      for (const move of legalMoves) {
         if (policyOutput[move] > maxPolicy) {
            maxPolicy = policyOutput[move];
            bestMove = move;
         }
      }

      if (bestMove !== -1) {
         gameBoard.applyMove(bestMove);
         renderBoard();
         updateGameInfo();
      } else {
         console.warn("AI could not find a legal move. Passing turn.");
         gameBoard.applyMove(-1);
         renderBoard();
         updateGameInfo();
      }
   }, 400);
}

resetButton.addEventListener("click", () => {
   gameBoard.reset();
   initGame();
});

initGame();
