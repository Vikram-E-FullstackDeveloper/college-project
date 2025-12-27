import os
import joblib

def save_models(clf_model, reg_model, encoders, scaler, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    clf_model.save_model(os.path.join(model_dir,"fertilizer_type_model.cbm"))
    reg_model.save_model(os.path.join(model_dir,"fertilizer_quantity_model.cbm"))
    joblib.dump(encoders, os.path.join(model_dir,"label_encoders.pkl"))
    joblib.dump(scaler, os.path.join(model_dir,"scaler.pkl"))
    print("\n✅ Models and artifacts saved successfully in 'models/' folder.")
