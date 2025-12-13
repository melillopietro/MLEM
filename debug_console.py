import pandas as pd
import numpy as np
import os
import joblib
import json
import sys
import time
import subprocess

# === CONFIGURAZIONE COLORI ===
class Colors:
    HEADER = '\033[95m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_step(msg): print(f"\n{Colors.HEADER}{Colors.BOLD}=== {msg} ==={Colors.ENDC}")
def print_ok(msg): print(f"   {Colors.OKGREEN}✅ {msg}{Colors.ENDC}")
def print_err(msg): print(f"   {Colors.FAIL}❌ {msg}{Colors.ENDC}")
def print_warn(msg): print(f"   {Colors.WARNING}⚠️ {msg}{Colors.ENDC}")

# === UTILS ESECUZIONE ===
def run_module(script_name, args=[]):
    if not os.path.exists(script_name): return False
    cmd = [sys.executable, script_name] + args
    env = os.environ.copy(); env["PYTHONIOENCODING"] = "utf-8"
    try:
        subprocess.run(cmd, check=True, env=env)
        return True
    except: return False

print(f"{Colors.BOLD}🛡️  MLEM SYSTEM DIAGNOSTICS (FULL SUITE) 🛡️{Colors.ENDC}")

# === 1. FILE SYSTEM ===
print_step("FASE 1: INTEGRITÀ FILE")
files = ["dashboard.py", "training_manager.py", "dataset_ML_Formatter.py"]
missing = [f for f in files if not os.path.exists(f)]
if missing: print_err(f"Mancano: {missing}")
else: print_ok("Script core presenti.")

# === 2. SIMULAZIONE TRAINING COMPLETO ===
print_step("FASE 2: TEST TRAINING SUITE COMPLETA")
print("Lancio training 'Fast' (pochi alberi) per verificare tutti i modelli...")

# Usiamo pochi alberi/iterazioni per fare veloce ma testare che il codice giri per tutti i modelli
args = ["--n_estimators", "10", "--max_depth", "3"]
success = run_module("training_manager.py", args)

if success:
    print_ok("Script training eseguito.")
    
    # VERIFICA RISULTATI
    if os.path.exists("model_comparison_results_final.csv"):
        df = pd.read_csv("model_comparison_results_final.csv")
        models_found = df['model'].unique()
        print(f"   -> Modelli trovati nel report: {models_found}")
        
        if len(models_found) > 2:
            print_ok(f"Suite Completa confermata! ({len(models_found)} modelli)")
        else:
            print_warn(f"Trovati solo {len(models_found)} modelli. Verifica training_manager.py")
    else:
        print_err("CSV Risultati non generato.")
else:
    print_err("Training crashato.")

# === 3. TEST INVESTIGATORE ===
print_step("FASE 3: INVESTIGATORE (Logica N/D)")
if os.path.exists("XGBoost_best_model.pkl"):
    try:
        model = joblib.load("XGBoost_best_model.pkl")
        le = joblib.load("label_encoder.pkl")
        with open("features_config.json", 'r') as f: feat = json.load(f)
        
        # Test N/D
        vec = pd.DataFrame(0, index=[0], columns=feat)
        # Accendiamo una feature a caso (es la prima TTP che troviamo)
        ttp = next((c for c in feat if "T" in c), None)
        if ttp: vec[ttp] = 1
        
        probs = model.predict_proba(vec)[0]
        conf = max(probs) * 100
        print(f"   -> Test Input Scarso: Confidenza {conf:.2f}%")
        
        if conf < 60: print_ok("Logica N/D attiva (Confidenza bassa).")
        else: print_warn("Il modello è confidente anche con 1 feature.")
        
    except Exception as e: print_err(f"Errore test modello: {e}")
else:
    print_err("Modello XGBoost mancante.")

print_step("FINE DIAGNOSTICA")