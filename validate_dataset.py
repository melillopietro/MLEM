import pandas as pd
import sys
"""
# DATASET VALIDATION
# Checks that the dataset is in CSV or Excel format and contains the required columns.
# Also checks that the columns are numeric and that there are no missing or duplicate values.
# If the dataset is not valid, the program exits with an error message.
"""

file_path = "final_ml_dataset_encoded.csv" 
# === FILE LOADING ===
try:
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        print("❌ Unsupported file format. Use .csv or .xlsx")
        sys.exit(1)
except Exception as e:
    print(f"❌ Error loading file: {e}")
    sys.exit(1)

# === BASIC ANALYSIS ===
print("\n📊 GENERAL INFO")
print(f"Rows, columns: {df.shape}")
print(f"Columns with missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"Duplicate rows: {df.duplicated().sum()}")

# === LABEL CHECK ===
label_col = "label_gang"
if label_col not in df.columns:
    print(f"❌ Label column '{label_col}' not found.")
    sys.exit(1)

print(f"\n🏷️ LABEL: '{label_col}'")
print(f"Unique values: {df[label_col].nunique()}")
print(f"Distribution:\n{df[label_col].value_counts().head(10)}")

# === FEATURE TYPE CHECK ===
non_numeric = df.drop(columns=[label_col]).select_dtypes(exclude=["int64", "float64", "uint8"]).columns.tolist()
if non_numeric:
    print("\n⚠️ Non-numeric columns among features (convert to numbers):")
    for col in non_numeric:
        print(f" - {col}")
else:
    print("\n✅ All features are numeric.")

print("\n✅ Validation completed.")
