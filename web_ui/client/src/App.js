import React, { useState, useEffect } from 'react';
import './App.css';

const API_URL = 'http://localhost:5000';

function App() {
  const [board, setBoard] = useState(Array(64).fill(0));
  const [legalMoves, setLegalMoves] = useState([]);
  const [currentPlayer, setCurrentPlayer] = useState(1);
  const [gameOver, setGameOver] = useState(false);
  const [winner, setWinner] = useState(null);
  const [aiThoughts, setAiThoughts] = useState([]);

  useEffect(() => {
    fetchNewGame();
  }, []);

  const fetchNewGame = async () => {
    const response = await fetch(`${API_URL}/api/new_game`);
    const data = await response.json();
    setBoard(data.board);
    setLegalMoves(data.legal_moves);
    setCurrentPlayer(data.current_player);
    setGameOver(data.game_over);
    setWinner(null);
    setAiThoughts([]);
  };

  const handleMove = async (move) => {
    if (!legalMoves.includes(move) || gameOver) {
      return;
    }

    const response = await fetch(`${API_URL}/api/move`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ move }),
    });

    const data = await response.json();
    setBoard(data.board);
    setLegalMoves(data.legal_moves);
    setCurrentPlayer(data.current_player);
    setGameOver(data.game_over);
    setWinner(data.winner);
    setAiThoughts(data.ai_thoughts || []);
  };

  const getHeatmapColor = (qValue) => {
    const normalizedQ = (qValue + 1) / 2; // Normalize Q value from [-1, 1] to [0, 1]
    const red = Math.round(255 * (1 - normalizedQ));
    const green = Math.round(255 * normalizedQ);
    return `rgb(${red}, ${green}, 0)`;
  };

  return (
    <div className="App">
      <h1>Othello AI Visualizer</h1>
      <div className="game-info">
        <p>Current Player: {currentPlayer === 1 ? 'Black' : 'White'}</p>
        {gameOver && (
          <p className="winner">
            Winner: {winner === 1 ? 'Black' : winner === 2 ? 'White' : 'Draw'}
          </p>
        )}
      </div>
      <div className="board">
        {board.flat().map((cell, index) => {
          const isLegal = legalMoves.includes(index);
          const thought = aiThoughts.find(t => t.move === index);
          const style = thought ? { backgroundColor: getHeatmapColor(thought.q_value) } : {};

          return (
            <div
              key={index}
              className={`cell ${isLegal ? 'legal' : ''}`}
              style={style}
              onClick={() => handleMove(index)}
            >
              {cell === 1 && <div className="piece black"></div>}
              {cell === 2 && <div className="piece white"></div>}
              {thought && <div className="thought-info">N: {thought.n_visits}<br/>Q: {thought.q_value.toFixed(2)}</div>}
            </div>
          );
        })}
      </div>
      <button onClick={fetchNewGame}>New Game</button>
    </div>
  );
}

export default App;