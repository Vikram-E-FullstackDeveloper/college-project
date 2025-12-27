import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE
from boruta import BorutaPy

def run_boruta(X_df, y, is_classification=True, sample_size=20000, random_state=42, max_features=10):
    n = len(X_df)
    if n > sample_size:
        sample_idx = np.random.RandomState(random_state).choice(n, sample_size, replace=False)
        X_sample = X_df.iloc[sample_idx].values
        y_sample = y.iloc[sample_idx].values
    else:
        X_sample = X_df.values
        y_sample = y.values

    if is_classification:
        estimator = RandomForestClassifier(n_jobs=-1, max_depth=5, random_state=random_state)
    else:
        estimator = RandomForestRegressor(n_jobs=-1, max_depth=5, random_state=random_state)

    boruta = BorutaPy(estimator, n_estimators='auto', random_state=random_state, verbose=0)
    try:
        boruta.fit(X_sample, y_sample)
        return X_df.columns[boruta.support_].tolist()
    except:
        estimator.fit(X_sample, y_sample)
        importances = estimator.feature_importances_
        idx = np.argsort(importances)[::-1]
        k = min(max_features, X_df.shape[1])
        return list(X_df.columns[idx[:k]])

def run_rfe(X_df, y, estimator, max_features=10):
    k = min(max_features, X_df.shape[1])
    rfe = RFE(estimator=estimator, n_features_to_select=k, step=1)
    rfe.fit(X_df.values, y.values)
    return X_df.columns[rfe.support_].tolist()
