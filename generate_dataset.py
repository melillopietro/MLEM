import pandas as pd
from sklearn.model_selection import train_test_split
import os

input_file = "final_ml_dataset_encoded.csv"
output_dir = "dataset_split"
min_samples = 10

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_file)
label_column = "label_gang"

value_counts = df[label_column].value_counts()
gangs_to_keep = value_counts[value_counts >= min_samples].index
df_filtered = df[df[label_column].isin(gangs_to_keep)]

# Split in 70-15-15 (train-val-test)
train_val_df, test_df = train_test_split(
    df_filtered,
    test_size=0.15,
    stratify=df_filtered[label_column],
    random_state=42
)

train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.176,
    stratify=train_val_df[label_column],
    random_state=42
)

# split label and features
def split_X_y(df):
    X = df.drop(columns=[label_column])
    y = df[label_column]
    return X, y

X_train, y_train = split_X_y(train_df)
X_val, y_val = split_X_y(val_df)
X_test, y_test = split_X_y(test_df)

#save 
X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
X_val.to_csv(f"{output_dir}/X_val.csv", index=False)
y_val.to_csv(f"{output_dir}/y_val.csv", index=False)
X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

print(f"✅ Dataset salvato in '{output_dir}/' con stratificazione e minimo {min_samples} sample per gang.")
print(f"Train: {len(X_train)} righe | Val: {len(X_val)} | Test: {len(X_test)}")
