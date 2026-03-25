import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# PATH SETUP (ROBUST)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

crop_path = os.path.join(BASE_DIR, "data", "processed", "clean_crop.csv")
fert_path = os.path.join(BASE_DIR, "data", "processed", "clean_fertilizer.csv")

model_dir = os.path.join(BASE_DIR, "models")
os.makedirs(model_dir, exist_ok=True)

print(" Crop Path:", crop_path)
print(" Fert Path:", fert_path)


# LOAD DATA

crop_df = pd.read_csv(crop_path)
fert_df = pd.read_csv(fert_path)


# DEBUG: CHECK COLUMNS

print("\n Fertilizer Dataset Columns:")
print(fert_df.columns)


# TRAIN FUNCTION

def train_and_evaluate(X, y, model_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"\n {model_name} RESULTS")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

    return model


# 1. CROP MODEL

X_crop = crop_df.drop("label", axis=1)
y_crop = crop_df["label"]

crop_model = train_and_evaluate(X_crop, y_crop, "Crop Model")

joblib.dump(crop_model, os.path.join(model_dir, "crop_model.pkl"))


# 2. SOIL MODEL 

required_soil_cols = [
    "temperature",
    "humidity",
    "moisture",
    "ph",
    "Organic_Carbon",
    "Electrical_Conductivity",
    "N", "P", "K"
]

#  Check missing columns
missing_cols = [col for col in required_soil_cols if col not in fert_df.columns]

if missing_cols:
    print(f"\n❌ Missing columns for Soil Model: {missing_cols}")
    print(" Fix your feature engineering notebook and re-save dataset!")
else:
    X_soil = fert_df[required_soil_cols]
    y_soil = fert_df["Soil_Type"]

    soil_model = train_and_evaluate(X_soil, y_soil, "Soil Model")

    joblib.dump(soil_model, os.path.join(model_dir, "soil_model.pkl"))


# 3. FERTILIZER MODEL

X_fert = fert_df[
    [
        "temperature",
        "humidity",
        "moisture",
        "N", "P", "K"
    ]
]

y_fert = fert_df["Recommended_Fertilizer"]

fert_model = train_and_evaluate(X_fert, y_fert, "Fertilizer Model")

joblib.dump(fert_model, os.path.join(model_dir, "fertilizer_model.pkl"))

print("\n All models trained and saved successfully!")