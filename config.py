TRAINING_DATA_DIR = './Database/training_data'
MODELS_DIR = './Database/models'

# Train
CURRENT_GENERATION_DATA_SUBDIR = '1G'
SELF_PLAY_MODEL_PATH = f'{MODELS_DIR}/Transformer/0G_initial.h5'
TRAINED_MODEL_SAVE_PATH = f'{MODELS_DIR}/Transformer/1G_trained.h5'
TRANSFORMER_MODEL_PATH = f'{MODELS_DIR}/Transformer/0G_initial.h5'

NUM_PARALLEL_GAMES = 8
SIMS_N = 1500
C_PUCT = 2.5
TOTAL_GAMES = 1600
TRAINING_HOURS = 0
SAVE_DATA_EVERY_N_GAMES = 100
VS_RANDOM = False
MCTS_PREDICT_BATCH_SIZE = 50

# trainModel
EPOCHS = 100
BATCH_SIZE = 20
learning_rate = 0.00008

# Inspect
INSPECT_GENERATION_SUBDIR = '19G'
NUMS_TO_SHOW = 5

# Review
Model_Path = f'{MODELS_DIR}/Transformer/1G_trained.h5'
R_SIMS_N = 4
Play_Games_Num = 100

# compare_models
NUM_GAMES_COMPARE = 10
COMPARE_SIMS_N = 10
Model1_Path = f'{MODELS_DIR}/ResNet/1G_07-03-25.h5'
Model2_Path = f'{MODELS_DIR}/Transformer/1G_trained.h5'
Model1_Name = "ResNet_1G"
Model2_Name = "Transformer_1G"



# val_loss:  2.412179946899414
# learning_rate: 0.0009641006559074151
# optimizer: Adam
# label_smoothing: 0.08284473485358701
# value_loss_weight: 0.7823081461391925