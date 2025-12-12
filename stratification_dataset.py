import pandas as pd
import os
import sys

# CONFIGURAZIONE
DATA_DIR = "./dataset_split"

def check_stratification():
    print("📊 Verifica Bilanciamento Classi (Stratificazione)...")
    
    # Costruzione percorsi corretti
    y_train_path = os.path.join(DATA_DIR, "y_train.csv")
    y_test_path = os.path.join(DATA_DIR, "y_test.csv")

    # Verifica esistenza file
    if not all(os.path.exists(p) for p in [y_train_path, y_test_path]):
        print(f"⚠️ Impossibile trovare i file CSV in '{DATA_DIR}'. Assicurati di aver eseguito generate_dataset.py")
        return

    # Caricamento
    try:
        y_train = pd.read_csv(y_train_path)
        y_test = pd.read_csv(y_test_path)

        # Calcolo distribuzioni
        train_dist = y_train['label_gang'].value_counts(normalize=True)
        test_dist = y_test['label_gang'].value_counts(normalize=True)
        
        # Confronto semplice (Differenza media delle percentuali)
        diff = (train_dist - test_dist).abs().mean()
        print(f"   -> Differenza media distribuzione Train vs Test: {diff:.5f}")
        
        if diff < 0.05:
            print("✅ Stratificazione OK: Le distribuzioni sono statisticamente simili.")
        else:
            print("⚠️ ATTENZIONE: Possibile sbilanciamento tra Train e Test.")
            
    except Exception as e:
        print(f"⚠️ Errore durante la lettura dei file: {e}")

if __name__ == "__main__":
    check_stratification()