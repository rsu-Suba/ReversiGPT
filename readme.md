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