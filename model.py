import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE
from boruta import BorutaPy
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, r2_score

# -------------------------
# Config
# -------------------------
RANDOM_STATE = 42
BORUTA_SAMPLE_SIZE = 20000
RFE_MAX_FEATURES = 10
CATBOOST_ITERS = 300
MODEL_DIR = "models"
JSON_FILE = "fertilizer_recommendationnew.json"

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# Load CSV helper
# -------------------------
def load_csv_try(names):
    for n in names:
        if os.path.exists(n):
            print(f"Loading: {n}")
            return pd.read_csv(n)
    raise FileNotFoundError(f"None of files found: {names}")

crop_df = load_csv_try(["cp.csv", "crop_production.csv"])
season_df = load_csv_try(["sc.csv", "seasonwise_crop.csv"])
df = pd.merge(crop_df, season_df, on="Season", how="inner")
print("Merged Data Shape:", df.shape)

# -------------------------
# Preprocessing
# -------------------------
df.fillna(df.median(numeric_only=True), inplace=True)
cat_cols = df.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "")

# Encode categorical columns
categorical_cols = ['State_Name','District_Name','Season','Crop','class']
encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

# -------------------------
# Targets
# -------------------------
y_type = df['class'].astype(int)
y_quantity = df['Production'].astype(float)
X = df.drop(columns=['class','Production'])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# -------------------------
# Feature selection helpers
# -------------------------
def run_boruta(X_df, y, is_classification=True, sample_size=BORUTA_SAMPLE_SIZE):
    n = len(X_df)
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(n, min(n, sample_size), replace=False)
    X_sample = X_df.iloc[sample_idx].values
    y_sample = y.iloc[sample_idx].values
    estimator = RandomForestClassifier(n_jobs=-1, max_depth=5, random_state=RANDOM_STATE) if is_classification \
                else RandomForestRegressor(n_jobs=-1, max_depth=5, random_state=RANDOM_STATE)
    boruta = BorutaPy(estimator, n_estimators='auto', random_state=RANDOM_STATE, verbose=0)
    try:
        boruta.fit(X_sample, y_sample)
        return X_df.columns[boruta.support_].tolist()
    except:
        estimator.fit(X_sample, y_sample)
        importances = estimator.feature_importances_
        idx = np.argsort(importances)[::-1]
        return list(X_df.columns[idx[:min(RFE_MAX_FEATURES, X_df.shape[1])]])

def run_rfe(X_df, y, estimator, max_features=RFE_MAX_FEATURES):
    k = min(max_features, X_df.shape[1])
    rfe = RFE(estimator=estimator, n_features_to_select=k, step=1)
    rfe.fit(X_df.values, y.values)
    return X_df.columns[rfe.support_].tolist()

# -------------------------
# Feature selection
# -------------------------
sel_clf = run_boruta(X, y_type, True)
sel_reg = run_boruta(X, y_quantity, False)

X_clf = X[sel_clf].copy()
X_reg = X[sel_reg].copy()

rfe_clf_sel = run_rfe(X_clf, y_type, RandomForestClassifier(n_jobs=-1, max_depth=5, random_state=RANDOM_STATE))
rfe_reg_sel = run_rfe(X_reg, y_quantity, RandomForestRegressor(n_jobs=-1, max_depth=5, random_state=RANDOM_STATE))

X_clf = X_clf[rfe_clf_sel].copy()
X_reg = X_reg[rfe_reg_sel].copy()

# -------------------------
# Train/Test split
# -------------------------
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_type, test_size=0.2, random_state=RANDOM_STATE, stratify=y_type
)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_quantity, test_size=0.2, random_state=RANDOM_STATE
)

# -------------------------
# Train CatBoost models
# -------------------------
clf_model = CatBoostClassifier(iterations=CATBOOST_ITERS, depth=8, learning_rate=0.08,
                               random_seed=RANDOM_STATE, verbose=0)
clf_model.fit(X_train_clf, y_train_clf)

reg_model = CatBoostRegressor(iterations=CATBOOST_ITERS, depth=8, learning_rate=0.08,
                              random_seed=RANDOM_STATE, verbose=0)
reg_model.fit(X_train_reg, y_train_reg)

# -------------------------
# Evaluate
# -------------------------
print("Classification Accuracy:", accuracy_score(y_test_clf, clf_model.predict(X_test_clf)))
print("Regression R²:", r2_score(y_test_reg, reg_model.predict(X_test_reg)))

# -------------------------
# Save models and encoders
# -------------------------
clf_model.save_model(os.path.join(MODEL_DIR,"fertilizer_type_model.cbm"))
reg_model.save_model(os.path.join(MODEL_DIR,"fertilizer_quantity_model.cbm"))
joblib.dump(encoders, os.path.join(MODEL_DIR,"label_encoders.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR,"scaler.pkl"))
print("\n✅ Models and artifacts saved successfully in 'models/' folder.")

# -------------------------
# Load JSON fertilizer file
# -------------------------
with open(JSON_FILE, "r") as f:
    fert_json = json.load(f)

# -------------------------
# Example prediction + JSON recommendation
# -------------------------
# Using the first row from X_test_clf as demo
sample_input = X_test_clf.iloc[[0]]
fert_type_pred = clf_model.predict(sample_input)[0]
fert_qty_pred = reg_model.predict(sample_input)[0]

# Decode class if encoder exists
if 'class' in encoders:
    fert_type_pred = encoders['class'].inverse_transform([int(fert_type_pred)])[0]

# Show JSON recommendation for the crop if exists
for crop_name, crop_data in fert_json.items():
    if crop_name.lower() == fert_type_pred.lower():  # match predicted type with JSON crop
        print(f"\nRecommended Fertilizer Details for {crop_name}:")
        nutrients = crop_data.get("Nutrients", {})
        for nutrient, details in nutrients.items():
            print(f"{nutrient}: Fertilizer={details.get('Fertilizer','')}, "
                  f"Amount={details.get('Recommended_Amount_kg_per_hectare',0)}, "
                  f"Timing={details.get('Application_Timing','')}")
        additional = crop_data.get("Additional_Recommendations", {})
        if additional:
            print("\nAdditional Recommendations:")
            for k,v in additional.items():
                print(f"{k}: {v}")
