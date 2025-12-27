import pandas as pd
import json
import os

# -------------------------
# Config
# -------------------------
CSV_FILE = "cp.csv"                  # Input CSV file
CROP_COLUMN = "Crop"                 # Column containing crop names
OUTPUT_JSON = "fertilizer_recommendation.json"

# Ensure file exists
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"{CSV_FILE} not found!")

# -------------------------
# Load CSV and extract crops
# -------------------------
df = pd.read_csv(CSV_FILE)

if CROP_COLUMN not in df.columns:
    raise KeyError(f"Column '{CROP_COLUMN}' not found in CSV!")

crops = df[CROP_COLUMN].dropna().unique()
print(f"Found {len(crops)} unique crops.")

# -------------------------
# Create fertilizer JSON template
# -------------------------
fertilizer_json = {}

for crop in crops:
    fertilizer_json[crop] = {
        "Nutrients": {
            "Nitrogen": {
                "Fertilizer": "",
                "Recommended_Amount_kg_per_hectare": 0,
                "Application_Timing": ""
            },
            "Phosphorus": {
                "Fertilizer": "",
                "Recommended_Amount_kg_per_hectare": 0,
                "Application_Timing": ""
            },
            "Potassium": {
                "Fertilizer": "",
                "Recommended_Amount_kg_per_hectare": 0,
                "Application_Timing": ""
            },
            "Organic_Manure": {
                "Fertilizer": "",
                "Recommended_Amount_kg_per_hectare": 0,
                "Application_Timing": ""
            }
        },
        "Additional_Recommendations": {
            "Soil_pH": "",
            "Irrigation": "",
            "Spacing": "",
            "Remarks": ""
        }
    }

# -------------------------
# Save to JSON
# -------------------------
with open(OUTPUT_JSON, "w") as f:
    json.dump(fertilizer_json, f, indent=4)

print(f"✅ Fertilizer recommendation JSON created: {OUTPUT_JSON}")
