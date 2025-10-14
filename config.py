TRAINING_DATA_DIR = './Database/training_data'
MODELS_DIR = './Database/models'

# Train
CURRENT_GENERATION_DATA_SUBDIR = '2G'
SELF_PLAY_MODEL_PATH = f'{MODELS_DIR}/Transformer/1G_10-10-25.h5'
TRAINED_MODEL_SAVE_PATH = f'{MODELS_DIR}/Transformer/2G_10-15-25.h5'
TRANSFORMER_MODEL_PATH = f'{MODELS_DIR}/Transformer/1G_10-10-25.h5'

NUM_PARALLEL_GAMES = 6
SIMS_N = 1500
C_PUCT = 2.5
TOTAL_GAMES = 2000
TRAINING_HOURS = 0
SAVE_DATA_EVERY_N_GAMES = 100
VS_RANDOM = False
MCTS_PREDICT_BATCH_SIZE = 64

# trainModel
EPOCHS = 100
BATCH_SIZE = 64
learning_rate = 0.00029642264945523635

# Inspect
INSPECT_GENERATION_SUBDIR = '20G'
NUMS_TO_SHOW = 5

# Review
Model_Path = f'{MODELS_DIR}/Transformer/1G_10-10-25.h5'
R_SIMS_N = 2
Play_Games_Num = 100

# compare_models
NUM_GAMES_COMPARE = 10
COMPARE_SIMS_N = 2
Model1_Path = f'{MODELS_DIR}/ResNet/13G_07-21-25.h5'
Model2_Path = f'{MODELS_DIR}/Transformer/1G_10-10-25.h5'
Model1_Name = "ResNet_1G"
Model2_Name = "Transformer_1G"



# val_loss: 2.240459442138672
# learning_rate: 0.00029642264945523635
# optimizer: AdamW
# label_smoothing: 0.026782207663537425
# value_loss_weight: 1.0635380642004533