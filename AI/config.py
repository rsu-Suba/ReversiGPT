TRAINING_DATA_DIR = './data'
MODELS_DIR = './models'

# Train
CURRENT_GENERATION_DATA_SUBDIR = 'M2'
SELF_PLAY_MODEL_PATH = f'{MODELS_DIR}/TF/backup/MoE-2.keras'
TRAINED_MODEL_SAVE_PATH = f'{MODELS_DIR}/TF/Nano-v2.keras'
TRANSFORMER_MODEL_PATH = f'{MODELS_DIR}/TF/0G.h5'

NUM_PARALLEL_GAMES = 40
SIMS_N = 800
Model2_Path = f'{MODELS_DIR}/TF/MoE-2.keras'
C_PUCT = 2.0
TOTAL_GAMES = 10000
TRAINING_HOURS = 0
SAVE_DATA_EVERY_N_GAMES = 500
VS_RANDOM = False
MCTS_PREDICT_BATCH_SIZE = 40

# trainModel
EPOCHS = 100 # 100
BATCH_SIZE = 256
# learning_rate = 5.0e-5

# Inspect
INSPECT_GENERATION_SUBDIR = 'micro_test'
NUMS_TO_SHOW = 5

# Review
# Model_Path = f'{MODELS_DIR}/RN/19G.h5'
Model_Path = f'{MODELS_DIR}/TF/MoE-2.keras'
R_SIMS_N = 3
Play_Games_Num = 10

# compare_models
NUM_GAMES_COMPARE = 10
COMPARE_SIMS_N = 5
Model1_Path = f'{MODELS_DIR}/TF/MoE-1.h5'
Model2_Path = f'{MODELS_DIR}/TF/MoE-2.keras'
Model1_Name = "M1"
Model2_Name = "M2"



# optimizer: AdamW
# learning_rate: 6.018790701331087e-05
# label_smoothing: 0.04165291567423903