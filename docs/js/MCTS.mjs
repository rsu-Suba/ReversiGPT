import { OthelloBoard } from "./OthelloBoard.mjs";

class MCTSNode {
    constructor(gameBoard, player, parent = null, move = null, prior = 0.0) {
        this.gameBoard = gameBoard; // OthelloBoard instance (cloned)
        this.player = player;
        this.parent = parent;
        this.move = move;
        this.prior = prior;
        this.children = {};
        this.visits = 0;
        this.valueSum = 0.0;
        this.isGameOver = gameBoard.isGameOver();
        this.winner = this.isGameOver ? gameBoard.getWinner() : null;
    }

    get qValue() {
        return this.visits === 0 ? 0 : this.valueSum / this.visits;
    }

    ucbScore(cPuct) {
        if (!this.parent) return 0;
        const u = cPuct * this.prior * Math.sqrt(this.parent.visits) / (1 + this.visits);
        return -this.qValue + u;
    }

    selectChild(cPuct) {
        let bestChild = null;
        let bestScore = -Infinity;

        for (const move in this.children) {
            const child = this.children[move];
            const score = child.ucbScore(cPuct);
            if (score > bestScore) {
                bestScore = score;
                bestChild = child;
            }
        }
        return bestChild;
    }

    update(value) {
        this.visits++;
        this.valueSum += value;
    }
}

export class MCTS {
    constructor(model, cPuct = 2.0) {
        this.model = model;
        this.cPuct = cPuct;
    }

    async search(rootBoard, sims = 50) {
        const rootPlayer = rootBoard.currentPlayer;
        const root = new MCTSNode(rootBoard.clone(), rootPlayer);

        // Initial expansion
        await this.expand(root);

        for (let i = 0; i < sims; i++) {
            let node = root;
            const path = [node];

            // Selection
            while (Object.keys(node.children).length > 0 && !node.isGameOver) {
                node = node.selectChild(this.cPuct);
                path.push(node);
            }

            // Expansion & Evaluation
            let value = 0;
            if (!node.isGameOver) {
                // If the node has legal moves but no children, expand it
                // Note: expand() also evaluates using the model
                value = await this.expand(node);
            } else {
                // Game over terminal value
                if (node.winner === 0) value = 0;
                else value = (node.winner === node.player) ? 1 : -1;
            }

            // Backpropagation
            // Value is from the perspective of 'node.player'
            for (let j = path.length - 1; j >= 0; j--) {
                const pathNode = path[j];
                // If pathNode.player == node.player, add value.
                // Else (opponent), add -value.
                const v = (pathNode.player === node.player) ? value : -value;
                pathNode.update(v);
            }
            
            // Yield to UI every few iterations to prevent freezing
            if (i % 10 === 0) await new Promise(r => setTimeout(r, 0));
        }

        return root;
    }

    async expand(node) {
        const board = node.gameBoard;
        
        // Predict
        const input = tf.tensor4d([board.boardToInputPlanes()], [1, 8, 8, 2], "float32");
        const predictions = this.model.predict(input);
        
        let policy, value;
        // Handle various output formats (Tensor, Array, NamedTensorMap)
        // Assuming [Policy(64), Value(1)] based on recent conversions
        if (Array.isArray(predictions)) {
             policy = await predictions[0].data();
             value = (await predictions[1].data())[0];
             predictions.forEach(t => t.dispose());
        } else {
             // Fallback if model output is different structure
             // ... simplify for now assuming array
             const outputs = Object.values(predictions);
             // Usually output names are generic, verify by size
             let pTensor, vTensor;
             for(let t of outputs) {
                 if (t.size === 64) pTensor = t;
                 if (t.size === 1) vTensor = t;
             }
             policy = await pTensor.data();
             value = (await vTensor.data())[0];
             outputs.forEach(t => t.dispose());
        }
        input.dispose();

        const legalMoves = board.getLegalMoves();
        
        // Softmax/Filter policy for legal moves
        let policySum = 0;
        const validProbs = {};
        
        if (legalMoves.length > 0) {
            for (const move of legalMoves) {
                const p = policy[move];
                validProbs[move] = p;
                policySum += p;
            }
            
            // Normalize and Expand
            for (const move of legalMoves) {
                const prob = policySum > 0 ? validProbs[move] / policySum : 1.0 / legalMoves.length;
                
                const nextBoard = board.clone();
                nextBoard.applyMove(move);
                
                const child = new MCTSNode(nextBoard, nextBoard.currentPlayer, node, move, prob);
                node.children[move] = child;
            }
        } else {
            // Pass turn if no legal moves but game not over
            // Wait, OthelloBoard.isGameOver() checks if both passed? 
            // board.getLegalMoves() returns [] if no moves.
            // If game is not over, it means the other player might have moves, or this player passes.
            // OthelloBoard implementation handles pass internally or we need explicit pass?
            // Let's assume explicit pass if legalMoves is empty but !isGameOver.
            // Actually OthelloBoard logic: getLegalMoves returns empty list -> must pass.
            // In AlphaZero, Pass is often handled as a move or by applying pass to board.
            // OthelloBoard.applyMove(-1) is pass.
            
            const nextBoard = board.clone();
            nextBoard.applyMove(-1);
            // Note: If nextBoard state is same as current (both pass), isGameOver becomes true.
            
            // We treat pass as a single child
            const child = new MCTSNode(nextBoard, nextBoard.currentPlayer, node, -1, 1.0);
            node.children[-1] = child;
        }

        return value;
    }
}
