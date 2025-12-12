
import pandas as pd
import time
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb

"""
Loads the data.

Converts labels (label_gang) to numbers.

Try different classification algorithms.

Use Grid Search to find the best parameters.

Evaluate performance (F1 score and accuracy).

Save the best models.

Save the results in a CSV file

"""

# === CONFIG ===
DATA_DIR = "./dataset_split"
LABEL_MAPPING_OUTPUT = "label_mapping.csv"

# === LOAD DATA ===
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_val = pd.read_csv(os.path.join(DATA_DIR, "X_val.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
y_val = pd.read_csv(os.path.join(DATA_DIR, "y_val.csv"))

# === LABEL ENCODING ===
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train["label_gang"])
y_val_enc = le.transform(y_val["label_gang"])

# Save label mapping
pd.DataFrame({"encoded_label": range(len(le.classes_)), "gang": le.classes_}).to_csv(LABEL_MAPPING_OUTPUT, index=False)

# === BASE MODELS ===
models = {
    "XGBoost": xgb.XGBClassifier(objective='multi:softmax', num_class=len(le.classes_), use_label_encoder=False, eval_metric='mlogloss'),
    "RandomForest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "LogisticRegression": LogisticRegression(class_weight='balanced', max_iter=1000),
    "MLPClassifier": MLPClassifier(max_iter=300, early_stopping=True, random_state=42),
    "SVC": SVC(class_weight='balanced', probability=True),
    "KNN": KNeighborsClassifier(),
    "LightGBM": lgb.LGBMClassifier(
        objective='multiclass',
        num_class=len(le.classes_),
        min_data_in_leaf=10,
        min_split_gain=1e-3,
        max_depth=10,
        verbosity=-1,
        n_jobs=-1
    )
}

# === HYPERPARAMETER GRIDS ===
param_grids = {
    "RandomForest": {
        "n_estimators": [100, 300],
        "max_depth": [10, 15],
        "min_samples_leaf": [1, 3],
        "max_features": ['sqrt']
    },
    "LogisticRegression": {
        "C": [0.1, 1, 10],
        "solver": ['lbfgs']
    },
    "MLPClassifier": {
        "hidden_layer_sizes": [(100,), (100, 50)],
        "activation": ['relu', 'tanh']
    },
    "SVC": {
        "C": [0.1, 1, 10],
        "kernel": ['rbf', 'linear']
    },
    "KNN": {
        "n_neighbors": [3, 5, 7]
    },
    "XGBoost": {
        "n_estimators": [100, 300],
        "max_depth": [6, 10],
        "learning_rate": [0.05, 0.1]
    },
    "LightGBM": {
        "n_estimators": [100],
        "max_depth": [6],
        "learning_rate": [0.1]
    }
}

# === GRID SEARCH & TRAINING ===
best_models = {}
results = []

for name, model in models.items():
    print(f"🔍 Grid Search: {name}")
    grid = GridSearchCV(model, param_grids[name], cv=3, scoring='f1_macro', n_jobs=-1)
    start = time.time()
    grid.fit(X_train, y_train_enc)
    train_time = time.time() - start

    best_model = grid.best_estimator_
    best_models[name] = best_model

    print(f"✅ Best params for {name}: {grid.best_params_}")
    y_pred = best_model.predict(X_val)
    f1 = f1_score(y_val_enc, y_pred, average='macro')
    acc = np.mean(y_pred == y_val_enc)

    print(f"📈 {name} - F1 macro: {f1:.4f}, Accuracy: {acc:.4f}, Time: {train_time:.2f}s")
    results.append({
        "model": name,
        "f1_macro": f1,
        "accuracy": acc,
        "train_time_sec": train_time
    })

    joblib.dump(best_model, f"{name}_best_model.pkl")

# === SAVE RESULTS ===
df_results = pd.DataFrame(results)
df_results.to_csv("model_comparison_results.csv", index=False)
print("\n📊 Risultati salvati in model_comparison_results.csv")
