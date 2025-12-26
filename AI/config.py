TRAINING_DATA_DIR = './data'
MODELS_DIR = './models'

# Train
CURRENT_GENERATION_DATA_SUBDIR = '3G'
SELF_PLAY_MODEL_PATH = f'{MODELS_DIR}/TF/MoE-1.h5'
TRAINED_MODEL_SAVE_PATH = f'{MODELS_DIR}/TF/MoE-1-dynamic-test.h5'
TRANSFORMER_MODEL_PATH = f'{MODELS_DIR}/TF/0G.h5'

NUM_PARALLEL_GAMES = 30
SIMS_N = 600
C_PUCT = 1.41
TOTAL_GAMES = 5000
TRAINING_HOURS = 0
SAVE_DATA_EVERY_N_GAMES = 500
VS_RANDOM = False
MCTS_PREDICT_BATCH_SIZE = 100

# trainModel
EPOCHS = 100
learning_rate = 6.018790701331087e-05

# Inspect
INSPECT_GENERATION_SUBDIR = 'micro_test'
NUMS_TO_SHOW = 5

# Review
# Model_Path = f'{MODELS_DIR}/RN/19G.h5'
Model_Path = f'{MODELS_DIR}/TF/MoE-1-optuna.h5'
R_SIMS_N = 5
Play_Games_Num = 100

# compare_models
NUM_GAMES_COMPARE = 100
COMPARE_SIMS_N = 3
Model1_Path = f'{MODELS_DIR}/TF/MoE-1-optuna.h5'
Model2_Path = f'{MODELS_DIR}/TF/MoE-1-micro-test-dynamic.h5'
Model1_Name = "M1"
Model2_Name = "mt1"



# optimizer: AdamW
# learning_rate: 6.018790701331087e-05
# label_smoothing: 0.04165291567423903