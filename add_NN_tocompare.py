import pandas as pd
import joblib
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

# === CONFIGURAZIONE ===
DATA_DIR = "./dataset_split"
RESULTS_CSV = "model_comparison_results_final.csv" # FILE CORRETTO
NN_MODEL_PATH = "FFNN_model.keras" 

def add_nn_results():
    print("🧠 Aggiunta Risultati Deep Learning al confronto...")

    # 1. Verifica Esistenza Modello
    if not os.path.exists(NN_MODEL_PATH):
        print(f"❌ Errore: Modello {NN_MODEL_PATH} non trovato. Esegui prima NN_new.py")
        return

    # 2. Caricamento Dati
    try:
        X_val = pd.read_csv(os.path.join(DATA_DIR, "X_val.csv"))
        y_val = pd.read_csv(os.path.join(DATA_DIR, "y_val.csv"))
        y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")) # Per LabelEncoder
    except FileNotFoundError:
        print("❌ Errore: File dataset mancanti.")
        return

    # 3. Label Encoding (Fondamentale per Keras)
    le = LabelEncoder()
    le.fit(y_train["label_gang"])
    y_val_enc = le.transform(y_val["label_gang"])

    # 4. Predizione con Keras
    try:
        model = tf.keras.models.load_model(NN_MODEL_PATH)
        y_pred_prob = model.predict(X_val, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)

        f1 = f1_score(y_val_enc, y_pred, average='macro')
        acc = accuracy_score(y_val_enc, y_pred)
        
        print(f"   ✅ Keras FFNN: F1={f1:.4f}, Accuracy={acc:.4f}")

        # 5. Aggiornamento CSV Risultati
        if os.path.exists(RESULTS_CSV):
            df = pd.read_csv(RESULTS_CSV)
            
            # Rimuovi vecchia riga se esiste già per evitare duplicati
            df = df[df['model'] != 'Keras FFNN']
            
            # Crea nuova riga
            new_row = pd.DataFrame([{
                "model": "Keras FFNN",
                "f1_macro": f1,
                "accuracy": acc,
                "train_time_sec": 0.0, # Tempo non tracciato qui, mettiamo 0
                "mode": "full" # Assumiamo full se non specificato
            }])
            
            # Concatenazione sicura
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(RESULTS_CSV, index=False)
            print(f"   💾 Aggiunto a {RESULTS_CSV}")
        else:
            print(f"⚠️ File {RESULTS_CSV} non trovato. Creo nuovo.")
            pd.DataFrame([{
                "model": "Keras FFNN", "f1_macro": f1, "accuracy": acc, 
                "train_time_sec": 0.0, "mode": "full"
            }]).to_csv(RESULTS_CSV, index=False)

    except Exception as e:
        print(f"❌ Errore durante l'integrazione NN: {e}")

if __name__ == "__main__":
    add_nn_results()