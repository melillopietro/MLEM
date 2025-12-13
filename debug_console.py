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
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_step(msg): print(f"\n{Colors.HEADER}{Colors.BOLD}=== {msg} ==={Colors.ENDC}")
def print_sub(msg): print(f"{Colors.OKBLUE}   >> {msg}{Colors.ENDC}")
def print_ok(msg): print(f"   {Colors.OKGREEN}✅ {msg}{Colors.ENDC}")
def print_err(msg): print(f"   {Colors.FAIL}❌ {msg}{Colors.ENDC}")
def print_warn(msg): print(f"   {Colors.WARNING}⚠️ {msg}{Colors.ENDC}")

print(f"{Colors.BOLD}🛡️  MLEM ULTIMATE DIAGNOSTICS (MAPS & XAI READY) 🛡️{Colors.ENDC}")

# ==========================================
# 1. CONTROLLO LIBRERIE NUOVE
# ==========================================
print_step("FASE 1: CONTROLLO DIPENDENZE GRAFICHE")

# Check Plotly (Mappe)
try:
    import plotly
    import plotly.express as px
    print_ok(f"Plotly installato (v{plotly.__version__}) -> Mappe Abilitate.")
except ImportError:
    print_err("Plotly MANCANTE. La mappa nel Tab 2 non funzionerà.")

# Check SHAP (XAI)
try:
    import shap
    print_ok(f"SHAP installato (v{shap.__version__}) -> Explainable AI Abilitata.")
except ImportError:
    print_err("SHAP MANCANTE. I grafici a cascata nel Tab 4 non funzioneranno.")

# ==========================================
# 2. CONTROLLO DATI PER LA MAPPA
# ==========================================
print_step("FASE 2: VERIFICA DATI GEOGRAFICI (THREAT MAP)")
if os.path.exists("dataset_split/X_val.csv"):
    try:
        df_val = pd.read_csv("dataset_split/X_val.csv", nrows=5) # Leggiamo solo l'header
        country_cols = [c for c in df_val.columns if "country" in c]
        
        if len(country_cols) > 0:
            print_ok(f"Trovate {len(country_cols)} colonne geografiche (es. {country_cols[0]}).")
            print_ok("La Cyber Threat Map nel Tab 2 ha i dati per colorarsi.")
        else:
            print_warn("Nessuna colonna 'country' trovata. La mappa resterà grigia/vuota.")
    except Exception as e:
        print_err(f"Errore lettura dataset: {e}")
else:
    print_err("Dataset X_val.csv mancante. Esegui il Preprocessing!")

# ==========================================
# 3. CONTROLLO SUITE MODELLI
# ==========================================
print_step("FASE 3: CONTROLLO TRAINING COMPLETO")
if os.path.exists("model_comparison_results_final.csv"):
    df_res = pd.read_csv("model_comparison_results_final.csv")
    models = df_res['model'].unique()
    print_sub(f"Modelli trovati: {models}")
    
    if len(models) >= 5:
        print_ok(f"Suite Completa Presente ({len(models)} modelli).")
    elif len(models) > 0:
        print_warn(f"Training Parziale ({len(models)} modelli). Per la tesi meglio averne 5+.")
    else:
        print_err("File risultati vuoto.")
else:
    print_err("Nessun risultato trovato. Esegui il Training nel Tab 1.")

# ==========================================
# 4. TEST FORENSE & XAI ENGINE
# ==========================================
print_step("FASE 4: TEST INVESTIGATORE & SHAP ENGINE")

if os.path.exists("XGBoost_best_model.pkl") and os.path.exists("features_config.json"):
    try:
        # Caricamento
        model = joblib.load("XGBoost_best_model.pkl")
        with open("features_config.json", 'r') as f: feat = json.load(f)
        print_ok("Modello XGBoost caricato.")

        # Creazione Input Dummy (Simuliamo un attacco)
        input_vec = pd.DataFrame(0, index=[0], columns=feat)
        
        # Test Predizione Standard
        prob = model.predict_proba(input_vec)[0]
        conf = max(prob)
        print_ok(f"Inferenza standard funzionante (Confidenza test: {conf:.2%}).")

        # Test Motore SHAP (Critico!)
        print_sub("Test calcolo SHAP Values (Simulazione)...")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(input_vec)
            
            # Verifica che abbia prodotto numeri
            if np.any(shap_values.values):
                print_ok("Motore XAI Funzionante! Il grafico a cascata verrà generato.")
            else:
                print_warn("Motore XAI ha prodotto zeri. Potrebbe essere un caso limite.")
                
        except Exception as e:
            print_err(f"CRASH MOTORE SHAP: {e}")
            print("   -> L'app non si bloccherà (grazie al try-catch), ma niente grafico.")

    except Exception as e:
        print_err(f"Errore caricamento modello: {e}")
else:
    print_err("Modello mancante. Impossibile testare l'Investigatore.")

print_step("DIAGNOSTICA TERMINATA")