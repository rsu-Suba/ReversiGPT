import { OthelloBoard } from "./OthelloBoard.mjs";

class MCTSNode {
    constructor(gameBoard, player, parent = null, move = null, prior = 0.0) {
        this.gameBoard = gameBoard;
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

    async search(rootBoard, sims = 50, onProgress = null) {
        const rootPlayer = rootBoard.currentPlayer;
        const root = new MCTSNode(rootBoard.clone(), rootPlayer);

        await this.expand(root);
        for (let i = 0; i < sims; i++) {
            let node = root;
            const path = [node];
            while (Object.keys(node.children).length > 0 && !node.isGameOver) {
                node = node.selectChild(this.cPuct);
                path.push(node);
            }

            let value = 0;
            if (!node.isGameOver) {
                value = await this.expand(node);
            } else {
                if (node.winner === 0) value = 0;
                else value = (node.winner === node.player) ? 1 : -1;
            }

            for (let j = path.length - 1; j >= 0; j--) {
                const pathNode = path[j];
                const v = (pathNode.player === node.player) ? value : -value;
                pathNode.update(v);
            }
            
            if (onProgress) {
                onProgress(root);
            }
            if (i % 2 === 0) await new Promise(r => setTimeout(r, 0));
        }

        return root;
    }

    async expand(node) {
        const board = node.gameBoard;
        const input = tf.tensor4d([board.boardToInputPlanes()], [1, 8, 8, 2], "float32");
        const predictions = this.model.predict(input);
        
        let policy, value;
        if (Array.isArray(predictions)) {
             policy = await predictions[0].data();
             value = (await predictions[1].data())[0];
             predictions.forEach(t => t.dispose());
        } else {
             const outputs = Object.values(predictions);
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
        
        let policySum = 0;
        const validProbs = {};
        
        if (legalMoves.length > 0) {
            for (const move of legalMoves) {
                const p = policy[move];
                validProbs[move] = p;
                policySum += p;
            }
            
            for (const move of legalMoves) {
                const prob = policySum > 0 ? validProbs[move] / policySum : 1.0 / legalMoves.length;
                
                const nextBoard = board.clone();
                nextBoard.applyMove(move);
                
                const child = new MCTSNode(nextBoard, nextBoard.currentPlayer, node, move, prob);
                node.children[move] = child;
            }
        } else {
            const nextBoard = board.clone();
            nextBoard.applyMove(-1);
            const child = new MCTSNode(nextBoard, nextBoard.currentPlayer, node, -1, 1.0);
            node.children[-1] = child;
        }

        return value;
    }
}
