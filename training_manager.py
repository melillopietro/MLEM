import pandas as pd
import time
import os
import joblib
import numpy as np
import argparse
import sys
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score

# Import dei Classificatori
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Import condizionali per librerie esterne
try: import xgboost as xgb
except: xgb = None
try: import lightgbm as lgb
except: lgb = None

class Config:
    DATA_DIR = "./dataset_split"
    RESULTS_FILE = "model_comparison_results_final.csv"
    FEATURES_CONFIG_FILE = "features_config.json"
    ENCODER_FILE = "label_encoder.pkl"

def run_training(n_estimators=150, max_depth=15):
    print("📂 Caricamento e Pulizia Dataset...")
    try:
        X_train = pd.read_csv(os.path.join(Config.DATA_DIR, "X_train.csv"))
        X_val = pd.read_csv(os.path.join(Config.DATA_DIR, "X_val.csv"))
        y_train = pd.read_csv(os.path.join(Config.DATA_DIR, "y_train.csv"))
        y_val = pd.read_csv(os.path.join(Config.DATA_DIR, "y_val.csv"))
        
        # Pulizia colonne spazzatura
        drop = [c for c in X_train.columns if "Unnamed" in c or c.lower() in ["id", "index"]]
        if drop: X_train.drop(columns=drop, inplace=True); X_val.drop(columns=drop, inplace=True)
        
        # Salvataggio Configurazione Feature
        with open(Config.FEATURES_CONFIG_FILE, 'w') as f: json.dump(X_train.columns.tolist(), f)
        
        # Encoding Label
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train["label_gang"])
        y_val_enc = le.transform(y_val["label_gang"])
        joblib.dump(le, Config.ENCODER_FILE)
        
        results = []
        
        # === DEFINIZIONE MODELLI ===
        models_to_train = []

        # 1. XGBoost (Il Campione)
        if xgb:
            models_to_train.append({
                "name": "XGBoost",
                "pipeline": xgb.XGBClassifier(
                    objective='multi:softmax', num_class=len(le.classes_),
                    n_estimators=n_estimators, max_depth=max_depth, n_jobs=1
                ),
                "needs_scaling": False
            })

        # 2. RandomForest (Il Robusto)
        models_to_train.append({
            "name": "RandomForest",
            "pipeline": RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, n_jobs=1, class_weight='balanced'
            ),
            "needs_scaling": False
        })

        # 3. SVM (Support Vector Machine)
        models_to_train.append({
            "name": "SVM",
            "pipeline": SVC(kernel='linear', class_weight='balanced', probability=True),
            "needs_scaling": True # SVM richiede scaling
        })

        # 4. KNN (K-Nearest Neighbors)
        models_to_train.append({
            "name": "KNN",
            "pipeline": KNeighborsClassifier(n_neighbors=5, n_jobs=1),
            "needs_scaling": True
        })

        # 5. MLP (Multi-Layer Perceptron - Rete Neurale Base)
        models_to_train.append({
            "name": "NeuralNet_MLP",
            "pipeline": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42),
            "needs_scaling": True
        })

        # 6. LightGBM (Se installato)
        if lgb:
            models_to_train.append({
                "name": "LightGBM",
                "pipeline": lgb.LGBMClassifier(
                    n_estimators=n_estimators, max_depth=max_depth, 
                    objective='multiclass', num_class=len(le.classes_), n_jobs=1, verbose=-1
                ),
                "needs_scaling": False
            })

        # === CICLO DI TRAINING ===
        print(f"\n🚀 Avvio Training di {len(models_to_train)} modelli...")
        
        best_f1 = 0

        for m in models_to_train:
            print(f"   ⚙️  Training {m['name']}...", end=" ", flush=True)
            try:
                start = time.time()
                
                # Creazione Pipeline (Scaler + Modello) se serve
                if m["needs_scaling"]:
                    model = Pipeline([
                        ('scaler', StandardScaler()),
                        ('clf', m["pipeline"])
                    ])
                else:
                    model = m["pipeline"]
                
                # Fit
                model.fit(X_train, y_train_enc)
                dur = time.time() - start
                
                # Predict
                preds = model.predict(X_val)
                f1 = f1_score(y_val_enc, preds, average='macro')
                acc = accuracy_score(y_val_enc, preds)
                
                print(f"✅ F1: {f1:.4f} ({dur:.1f}s)")
                
                results.append({
                    "model": m["name"],
                    "f1_macro": f1,
                    "accuracy": acc,
                    "train_time_sec": dur
                })
                
                # Salviamo sempre XGBoost come default per la dashboard
                if m["name"] == "XGBoost":
                    joblib.dump(model, "XGBoost_best_model.pkl")
                
                if f1 > best_f1:
                    best_f1 = f1
                    
            except Exception as e:
                print(f"❌ Errore: {e}")

        # Salvataggio CSV Finale
        if results:
            df_res = pd.DataFrame(results)
            df_res.to_csv(Config.RESULTS_FILE, index=False)
            print(f"\n📊 Risultati salvati in {Config.RESULTS_FILE}")
        
    except Exception as e: print(f"❌ Errore Critico Pipeline: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=150)
    parser.add_argument("--max_depth", type=int, default=15)
    # Argomenti extra ignorati
    args, unknown = parser.parse_known_args()
    
    run_training(args.n_estimators, args.max_depth)