import os

RANDOM_STATE = 42
BORUTA_SAMPLE_SIZE = 20000
RFE_MAX_FEATURES = 10
CATBOOST_ITERS = 300
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)
