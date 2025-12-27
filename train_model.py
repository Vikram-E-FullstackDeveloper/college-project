

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from feature_selection import run_boruta, run_rfe

def train_models(X, y_type, y_quantity, random_state=42, catboost_iters=300):
    # Feature selection
    sel_clf = run_boruta(X, y_type, True, random_state=random_state)
    sel_reg = run_boruta(X, y_quantity, False, random_state=random_state)

    X_clf = X[sel_clf].copy()
    X_reg = X[sel_reg].copy()

    rfe_clf_sel = run_rfe(X_clf, y_type, RandomForestClassifier(n_jobs=-1, max_depth=5, random_state=random_state))
    rfe_reg_sel = run_rfe(X_reg, y_quantity, RandomForestRegressor(n_jobs=-1, max_depth=5, random_state=random_state))

    X_clf = X_clf[rfe_clf_sel].copy()
    X_reg = X_reg[rfe_reg_sel].copy()

    # Split
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_type, test_size=0.2, random_state=random_state, stratify=y_type
    )
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_quantity, test_size=0.2, random_state=random_state
    )

    # Train CatBoost
    clf_model = CatBoostClassifier(iterations=catboost_iters, depth=8, learning_rate=0.08,
                                   random_seed=random_state, verbose=0)
    clf_model.fit(X_train_clf, y_train_clf)

    reg_model = CatBoostRegressor(iterations=catboost_iters, depth=8, learning_rate=0.08,
                                  random_seed=random_state, verbose=0)
    reg_model.fit(X_train_reg, y_train_reg)

    # Evaluate
    print("Classification Accuracy:", accuracy_score(y_test_clf, clf_model.predict(X_test_clf)))
    print("Regression R²:", r2_score(y_test_reg, reg_model.predict(X_test_reg)))

    return clf_model, reg_model
