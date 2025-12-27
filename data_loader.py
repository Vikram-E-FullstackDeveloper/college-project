import os
import pandas as pd

def load_csv_try(names):
    for n in names:
        if os.path.exists(n):
            print(f"Loading: {n}")
            return pd.read_csv(n)
    raise FileNotFoundError(f"None of files found: {names}")

def load_crop_data():
    crop_df = load_csv_try(["cp.csv", "crop_production.csv"])
    season_df = load_csv_try(["sc.csv", "seasonwise_crop.csv"])
    df = pd.merge(crop_df, season_df, on="Season", how="inner")
    print("Merged Data Shape:", df.shape)
    return df
