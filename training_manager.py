import pandas as pd
import time
import os
import joblib
import numpy as np
import argparse
import sys
import re
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from scipy.stats import randint, uniform, loguniform

class Config:
    DATA_DIR = "./dataset_split"
    RESULTS_FILE = "model_comparison_results_final.csv"
    FEATURES_CONFIG_FILE = "features_config.json"
    CV_FOLDS = 3
    RANDOM_STATE = 42

def load_data():
    try:
        print("📂 Caricamento dataset...")
        X_train = pd.read_csv(os.path.join(Config.DATA_DIR, "X_train.csv"))
        X_val = pd.read_csv(os.path.join(Config.DATA_DIR, "X_val.csv"))
        y_train = pd.read_csv(os.path.join(Config.DATA_DIR, "y_train.csv"))
        y_val = pd.read_csv(os.path.join(Config.DATA_DIR, "y_val.csv"))

        # FILTRO TTP STANDARD
        all_cols = X_train.columns.tolist()
        ttp_pattern = re.compile(r"^T\d{4}") 
        selected_cols = [c for c in all_cols if ttp_pattern.match(c)]
        if not selected_cols: selected_cols = all_cols
        
        print(f"   -> Feature TTP selezionate: {len(selected_cols)}")
        
        X_train = X_train[selected_cols]
        X_val = X_val[selected_cols]
        
        with open(Config.FEATURES_CONFIG_FILE, 'w') as f:
            json.dump(selected_cols, f)
            
        return X_train, X_val, y_train, y_val
    except Exception as e:
        print(f"❌ Errore caricamento: {e}")
        return None

def run_training(n_estimators, max_depth, n_iter):
    data = load_data()
    if not data: return
    X_train, X_val, y_train, y_val = data

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train["label_gang"])
    y_val_enc = le.transform(y_val["label_gang"])
    
    print(f"\n⚙️  CONFIGURAZIONE SIMULAZIONE:")
    print(f"   - N. Alberi: {n_estimators}")
    print(f"   - Max Depth: {max_depth}")
    print(f"   - Iterazioni: {n_iter}")

    # CONFIGURAZIONE DINAMICA MODELLI
    # Usiamo i valori passati come LISTA [valore] per forzare quel parametro
    models = {
        "RandomForest": {
            "model": RandomForestClassifier(class_weight='balanced', n_jobs=1),
            "params": {
                "classifier__n_estimators": [n_estimators], 
                "classifier__max_depth": [max_depth]
            }
        },
        "SVC": {
            "model": SVC(class_weight='balanced', probability=False),
            "params": {"classifier__C": loguniform(0.1, 10), "classifier__kernel": ['linear']}
        },
        "KNN": {
             "model": KNeighborsClassifier(n_jobs=1),
             "params": {"classifier__n_neighbors": randint(3, 10)}
        }
    }
    
    try:
        import xgboost as xgb
        models["XGBoost"] = {
            "model": xgb.XGBClassifier(objective='multi:softmax', num_class=len(le.classes_), n_jobs=1),
            "params": {
                "classifier__n_estimators": [n_estimators],
                "classifier__max_depth": [max_depth]
            }
        }
    except: pass

    try:
        import lightgbm as lgb
        models["LightGBM"] = {
            "model": lgb.LGBMClassifier(objective='multiclass', num_class=len(le.classes_), verbosity=-1, n_jobs=1),
            "params": {
                "classifier__n_estimators": [n_estimators],
                "classifier__max_depth": [max_depth]
            }
        }
    except: pass

    results = []
    print(f"\n🚀 START TRAINING SIMULAZIONE...")

    for name, config in models.items():
        filename = f"{name}_best_model.pkl"
        print(f"🔥 TRAINING: {name}...", end=" ", flush=True)
        try:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', config["model"])
            ])
            
            # Se i parametri sono fissi (liste di 1 elemento), n_iter > 1 è inutile ma innocuo
            actual_iter = 1 if len(config["params"].get("classifier__n_estimators", [])) == 1 else n_iter

            search = RandomizedSearchCV(
                pipeline, config["params"], n_iter=actual_iter, 
                cv=Config.CV_FOLDS, scoring='f1_macro', n_jobs=1, 
                random_state=Config.RANDOM_STATE
            )
            
            start = time.time()
            search.fit(X_train, y_train_enc)
            duration = time.time() - start
            
            best = search.best_estimator_
            f1 = f1_score(y_val_enc, best.predict(X_val), average='macro')
            acc = accuracy_score(y_val_enc, best.predict(X_val))
            
            print(f"✅ F1: {f1:.4f} ({duration:.1f}s)")
            
            joblib.dump(best, filename)
            results.append({"model": name, "f1_macro": f1, "accuracy": acc, "train_time_sec": duration})
        except Exception as e:
            print(f"❌ Errore: {e}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(Config.RESULTS_FILE, index=False)
        print(f"📊 Risultati simulazione salvati.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Argomenti che arrivano dalla Dashboard
    parser.add_argument("--n_estimators", type=int, default=150)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--n_iter", type=int, default=5)
    
    args = parser.parse_args()
    
    run_training(args.n_estimators, args.max_depth, args.n_iter)