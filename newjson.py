import pandas as pd
import json

# === Step 1: Load Crop Field Data ===
df = pd.read_csv('cp.csv')

# === Step 2: Define Nutrient Assessment Logic ===
def assess_nutrient(value, low, high):
    if value < low:
        return "Low"
    elif value > high:
        return "High"
    else:
        return "Medium"

def recommend_fertilizers(n, p, k):
    recs = []
    if n == "Low":
        recs.append("Urea (N)")
    elif n == "High":
        recs.append("Avoid Urea (N)")
    
    if p == "Low":
        recs.append("SSP (P)")
    elif p == "High":
        recs.append("Avoid SSP (P)")
    
    if k == "Low":
        recs.append("MOP (K)")
    elif k == "High":
        recs.append("Avoid MOP (K)")
    
    return recs

def schedule(n_status):
    if n_status == "Low":
        return {
            "Basal": ["SSP - 50 kg/acre"],
            "Top Dressing": ["Urea - 60 kg/acre at vegetative", "Urea - 40 kg/acre at flowering"]
        }
    elif n_status == "Medium":
        return {
            "Basal": ["SSP - 40 kg/acre", "MOP - 20 kg/acre"],
            "Top Dressing": ["Urea - 50 kg/acre at mid-growth"]
        }
    else:
        return {
            "Basal": ["SSP - 30 kg/acre", "MOP - 15 kg/acre"],
            "Top Dressing": ["No additional nitrogen required"]
        }

# === Step 3: Build JSON Structure ===
recommendations = []

for _, row in df.iterrows():
    crop = row.get('Crop', 'Unknown')
    soil = row.get('Soil_Type', 'Unknown')
    pH = float(row.get('pH', 7.0))
    N = float(row.get('Nitrogen', 0))
    P = float(row.get('Phosphorus', 0))
    K = float(row.get('Potassium', 0))

    n_status = assess_nutrient(N, 50, 150)
    p_status = assess_nutrient(P, 20, 60)
    k_status = assess_nutrient(K, 40, 100)

    recommendation = {
        "crop": crop,
        "soil_type": soil,
        "pH": round(pH, 2),
        "nutrient_status": {
            "Nitrogen": n_status,
            "Phosphorus": p_status,
            "Potassium": k_status
        },
        "recommended_fertilizers": recommend_fertilizers(n_status, p_status, k_status),
        "application_schedule": schedule(n_status)
    }

    recommendations.append(recommendation)

# === Step 4: Save to JSON ===
output = {"recommendations": recommendations}

with open('fertilizer_recommendationnew.json', 'w') as f:
    json.dump(output, f, indent=2)

print("✅ Fertilizer recommendation JSON saved as 'fertilizer_recommendation.json'")
