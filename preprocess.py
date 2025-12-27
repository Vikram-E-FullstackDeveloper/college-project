import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess(df, categorical_cols=['State_Name','District_Name','Season','Crop','class']):
    # Fill numeric missing values
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    # Fill categorical missing values
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "")
    
    # Encode categorical
    encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # Targets
    y_type = df['class'].astype(int)
    y_quantity = df['Production'].astype(float)
    X = df.drop(columns=['class','Production'])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X, y_type, y_quantity, encoders, scaler
