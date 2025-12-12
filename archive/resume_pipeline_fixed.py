import pandas as pd
import time
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from scipy.stats import randint, uniform, loguniform

# === CONFIG ===
DATA_DIR = "./dataset_split"
RESULTS_OUTPUT = "model_comparison_results_final.csv"
N_ITER_SEARCH = 15 

def main():
    # === LOAD DATA ===
    print("📂 Caricamento dati...")
    try:
        X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
        X_val = pd.read_csv(os.path.join(DATA_DIR, "X_val.csv"))
        y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
        y_val = pd.read_csv(os.path.join(DATA_DIR, "y_val.csv"))
    except FileNotFoundError as e:
        print(f"❌ Errore: File non trovati in {DATA_DIR}. Assicurati di aver eseguito generate_dataset.py")
        return

    # === LABEL ENCODING ===
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train["label_gang"])
    y_val_enc = le.transform(y_val["label_gang"])

    results = []

    # === 1. RECUPERO MODELLI ESISTENTI (XGBoost & RF) ===
    print("\n♻️  Recupero modelli già addestrati...")
    existing_models = ["XGBoost", "RandomForest"]

    for name in existing_models:
        filename = f"{name}_best_model.pkl"
        if os.path.exists(filename):
            print(f"   -> Trovato {filename}, valuto le performance...")
            try:
                model = joblib.load(filename)
                
                # Gestione robusta per modelli o pipeline
                if hasattr(model, "predict"):
                    y_pred = model.predict(X_val)
                else:
                    print(f"      ⚠️ Oggetto {name} non valido, salto.")
                    continue
                
                f1 = f1_score(y_val_enc, y_pred, average='macro')
                acc = np.mean(y_pred == y_val_enc)
                print(f"      ✅ {name} recuperato: F1={f1:.4f}, Acc={acc:.4f}")
                results.append({
                    "model": name, "f1_macro": f1, "accuracy": acc, 
                    "train_time_sec": 0, "best_params": "Loaded from pkl"
                })
            except Exception as e:
                print(f"      ⚠️ Errore nel caricamento di {filename}: {e}")
        else:
            print(f"      ❌ {filename} non trovato (sarà ignorato se non previsto dopo).")

    # === 2. TRAINING NUOVI MODELLI ===
    models_to_run = {
        "MLPClassifier": {
            "model": MLPClassifier(max_iter=500, early_stopping=True, random_state=42),
            "params": {
                "classifier__hidden_layer_sizes": [(100,), (100, 50), (256, 128)],
                "classifier__activation": ['relu', 'tanh'],
                "classifier__alpha": loguniform(1e-5, 1e-2)
            }
        },
        "SVC": {
            "model": SVC(class_weight='balanced', probability=False, random_state=42),
            "params": {
                "classifier__C": loguniform(0.1, 100),
                "classifier__kernel": ['linear', 'rbf'], 
                "classifier__gamma": ['scale', 'auto']
            }
        },
        "KNN": {
            "model": KNeighborsClassifier(n_jobs=1), # Importante: n_jobs=1 qui per evitare conflitti interni
            "params": {
                "classifier__n_neighbors": randint(3, 15),
                "classifier__weights": ['uniform', 'distance']
            }
        },
        "LightGBM": {
            # n_jobs=1 nel learner per evitare deadlock, parallelismo gestito da RandomizedSearchCV
            "model": lgb.LGBMClassifier(objective='multiclass', num_class=len(le.classes_), verbosity=-1, n_jobs=1),
            "params": {
                "classifier__n_estimators": randint(100, 500),
                "classifier__learning_rate": loguniform(0.01, 0.3),
                "classifier__num_leaves": randint(20, 100),
                "classifier__max_depth": randint(-1, 20)
            }
        }
    }

    print(f"\n🚀 Avvio training modelli rimanenti...")

    for name, config in models_to_run.items():
        print(f"\n🔍 Tuning: {name}")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', config["model"])
        ])
        
        # Su Windows, n_jobs=-1 può essere pesante. Se crasha ancora, metti n_jobs=1.
        search = RandomizedSearchCV(
            pipeline, config["params"], n_iter=N_ITER_SEARCH, 
            cv=3, scoring='f1_macro', n_jobs=-1, random_state=42, verbose=1
        )
        
        start = time.time()
        try:
            search.fit(X_train, y_train_enc)
        except Exception as e:
            print(f"⚠️ Errore critico su {name}: {e}")
            continue
            
        train_time = time.time() - start
        best_model = search.best_estimator_
        
        # Valutazione
        y_pred = best_model.predict(X_val)
        f1 = f1_score(y_val_enc, y_pred, average='macro')
        acc = np.mean(y_pred == y_val_enc)

        print(f"✅ Best params: {search.best_params_}")
        print(f"📈 {name} - F1: {f1:.4f}, Acc: {acc:.4f}, Time: {train_time:.2f}s")
        
        results.append({
            "model": name, "f1_macro": f1, "accuracy": acc, 
            "train_time_sec": train_time, "best_params": str(search.best_params_)
        })

        # Salvataggio
        joblib.dump(best_model, f"{name}_best_model.pkl")

    # === SAVE FINAL RESULTS ===
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(RESULTS_OUTPUT, index=False)
        print(f"\n📊 Risultati completi salvati in {RESULTS_OUTPUT}")
    else:
        print("\n⚠️ Nessun risultato generato.")

if __name__ == "__main__":
    # Questo blocco è OBBLIGATORIO su Windows per il multiprocessing
    main()