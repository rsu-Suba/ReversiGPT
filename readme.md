## Commands

-  Study `python train.py` -> `python ./Database/tfrecord.py` -> `python ./Database/trainModel.py`
-  Review `python review.py` (AI vs Random Bot)
-  Review `python reviewHuman.py` (AI vs Human)

## Files used

-  train.py : Study
-  tfrecord.py : Convert msgpacks to TFRecord
-  trainModel.py : Create new AI model.h5 from TFRecord
-  review.py : Review (vs Random bot)
-  reviewHuman.py : Review (vs Human input)
-  config.py : Parameters file for all program

### Module files(Required)

-  reversi_bitboard_cpp.cpython-311-x86_64-linux-gnu.so : Compiled AI engine (C++)
-  ./cpp_reversi : AI engine (C++)

### Model Architecture
#### Iterative Dynamic Routing MoE Transformer
A specialized Transformer architecture designed for Othello, featuring dynamic computation paths.

- **Backbone**: GPT-like Decoder-only Transformer
- **Input Embedding**: 
  - Spatial (Row/Col) + Temporal (Move count) Embeddings
- **Dynamic Blocks**: 
  - Iterative Dynamic Routing (Recurrent processing within blocks)
  - Mixture of Experts (Selectable MHA & FFN modules)
  - Pre-LN with Residual connections
- **Dual-Head Output**:
  - **Policy Head**: Predicts the best move (Softmax)
  - **Value Head**: Evaluates board position (Tanh, Flattened feature map)
- **Source**: [`dynamic_MoE.py`](./AI/models/dynamic_MoE.py)

> Model hyperparameters (layers, heads, dimensions) can be configured in `model.yaml`.