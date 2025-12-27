from data_loader import load_crop_data
from preprocessing import preprocess
from train_model import train_models
from save_artifacts import save_models
from constants import RANDOM_STATE, CATBOOST_ITERS, MODEL_DIR

# Load data
df = load_crop_data()

# Preprocess
X, y_type, y_quantity, encoders, scaler = preprocess(df)

# Train models
clf_model, reg_model = train_models(X, y_type, y_quantity, RANDOM_STATE, CATBOOST_ITERS)

# Save artifacts
save_models(clf_model, reg_model, encoders, scaler, MODEL_DIR)
