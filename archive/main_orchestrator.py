import os
import subprocess
import sys

def run_script(script_name, description, critical=True):
    print(f"\n🔹 [STEP] {description} ({script_name})...")
    
    if not os.path.exists(script_name):
        print(f"⚠️ Script {script_name} non trovato. Salto.")
        if critical: sys.exit(1)
        return

    try:
        subprocess.run([sys.executable, script_name], check=True)
        print(f"✅ {script_name} completato.")
    except subprocess.CalledProcessError:
        if critical:
            print(f"❌ ERRORE CRITICO in {script_name}. Stop.")
            sys.exit(1)
        else:
            print(f"⚠️ {script_name} ha dato errore, ma continuo.")

def main():
    print("=== MLEM PIPELINE (FULL PHD AUTOMATION) ===\n")
    
    # 1. Dataset Preprocessing
    if not os.path.exists("Dataset Normalized.csv"):
        run_script("normalized_dataset.py", "Normalizzazione Dataset")
    
    if not os.path.exists("final_ml_dataset_encoded.csv"):
        run_script("dataset_ML_Formatter.py", "Encoding One-Hot")

    if not os.path.exists("dataset_split/X_train.csv"):
        run_script("generate_dataset.py", "Splitting Train/Test")
    else:
        print("⏩ SKIP: Dataset già splittato.")

    # 2. Check (Non bloccante)
    run_script("stratification_dataset.py", "Check Stratificazione", critical=False)

    # 3. Training (Il cuore del processo)
    # Questo script gestisce già il salto dei modelli esistenti
    run_script("training_manager.py", "Training e Ottimizzazione")

    # 4. Analisi & Grafici
    # Genera CSV dettagliati per classe + SHAP
    run_script("final_fixed.py", "Analisi Dettagliata Errori & SHAP", critical=False)
    
    # Genera i 4 Grafici PNG per il report
    run_script("generate_final_graphs.py", "Generazione Grafici (ROC/PR/CM)", critical=False)

    # 5. Scrittura Report (NUOVO STEP)
    # Legge i CSV e scrive il testo del report finale
    run_script("generate_automatic_report.py", "Scrittura Automatica Report PhD", critical=False)

    print("\n🎉 PIPELINE COMPLETATA! Il tuo esame è pronto.")

if __name__ == "__main__":
    main()