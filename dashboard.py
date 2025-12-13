import streamlit as st
import pandas as pd
import os
import subprocess
import sys
import time
import joblib
import json
import numpy as np

# === CONFIGURAZIONE ===
st.set_page_config(page_title="MLEM Dashboard", page_icon="🛡️", layout="wide")
RESULTS_CSV = "model_comparison_results_final.csv"
FEATURES_CONFIG = "features_config.json"
MODEL_FILE = "XGBoost_best_model.pkl"
ENCODER_FILE = "label_encoder.pkl"
DATA_DIR = "./dataset_split"
RAW_DATASET_XLSX = "Dataset Ransomware.xlsx"
RAW_DATASET_CSV = "Dataset Normalized.csv"

st.title("🛡️ MLEM: Ransomware Attribution Pipeline")

# === UTILS ===
def run_command(cmd_args, log_box):
    # Gestione robusta dei comandi sia stringa che lista
    if isinstance(cmd_args, str):
        cmd = [sys.executable, cmd_args]
    else:
        cmd = [sys.executable] + cmd_args
        
    env = os.environ.copy(); env["PYTHONIOENCODING"] = "utf-8"
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', env=env)
        while True:
            line = p.stdout.readline()
            if not line and p.poll() is not None: break
            if line: log_box.code(line.strip(), language="bash"); time.sleep(0.001)
        return p.returncode == 0
    except Exception as e: log_box.error(f"Err: {e}"); return False

# === SIDEBAR ===
with st.sidebar:
    st.header("🎛️ Pannello di Controllo")
    
    # 1. DATI
    st.markdown("### 1. Dati")
    source = st.radio("Sorgente:", ["Default", "Upload Locale"], label_visibility="collapsed")
    if source == "Upload Locale":
        up = st.file_uploader("Carica Excel/CSV", type=['xlsx', 'csv'])
        if up and st.button("Usa questo file"):
            dest = RAW_DATASET_XLSX if up.name.endswith('.xlsx') else RAW_DATASET_CSV
            with open(dest, "wb") as f: f.write(up.getbuffer())
            st.success("✅ Caricato!")
            
    st.divider()

    # 2. PARAMETRI SIMULAZIONE
    st.markdown("### 2. 🛠️ Parametri Modello")
    n_estimators = st.slider("Numero Alberi (Estimators)", 50, 500, 150, step=10, help="Più alberi = più stabilità, più tempo.")
    max_depth = st.slider("Profondità Max (Depth)", 3, 50, 15, step=1, help="Capacità di imparare pattern complessi.")
    
    st.divider()

    # 3. AZIONI (TUTTE LE FASI RIPRISTINATE)
    st.markdown("### 3. Fasi Pipeline")
    run_prep = st.checkbox("1. Preprocessing", value=False)
    run_train = st.checkbox("2. Training Modelli", value=True)
    run_anal = st.checkbox("3. Analisi & Report", value=True)
    
    st.divider()
    if st.button("🗑️ RESET FILES"):
        for f in [RESULTS_CSV, MODEL_FILE, ENCODER_FILE, FEATURES_CONFIG, "RandomForest_best_model.pkl"]:
            if os.path.exists(f): os.remove(f)
        st.warning("Cache pulita. Rilancia la pipeline.")

# === TABS ===
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Esecuzione", "📊 Risultati", "📥 Download", "🕵️ Investigatore"])

# --- TAB 1: ESECUZIONE ---
with tab1:
    if st.button("▶️ AVVIA PIPELINE COMPLETA", type="primary", use_container_width=True):
        
        # FASE 1: PREPROCESSING
        if run_prep:
            with st.status("Preprocessing...", expanded=False) as s:
                run_command("normalized_dataset.py", s)
                run_command("dataset_ML_Formatter.py", s)
                run_command("generate_dataset.py", s)
                run_command("stratification_dataset.py", s)
                s.update(label="Preprocessing OK", state="complete")

        # FASE 2: TRAINING (Con Parametri Slider)
        if run_train:
            with st.status(f"Training (Est: {n_estimators}, Depth: {max_depth})...", expanded=True) as s:
                # Passiamo i parametri degli slider allo script
                cmd = ["training_manager.py", "--n_estimators", str(n_estimators), "--max_depth", str(max_depth)]
                if run_command(cmd, s):
                    s.write("ML Training OK")
                
                # Eseguiamo reti neurali se presenti (come da tua richiesta)
                if os.path.exists("NN_new.py"):
                    try: 
                        import tensorflow
                        run_command("NN_new.py", s)
                    except: pass
                
                if os.path.exists("add_NN_tocompare.py"):
                    run_command("add_NN_tocompare.py", s)
                
                s.update(label="Training Completato", state="complete")

        # FASE 3: ANALISI
        if run_anal:
            with st.status("Generazione Report...", expanded=False) as s:
                if os.path.exists("final_fixed.py"): run_command("final_fixed.py", s)
                if os.path.exists("generate_final_graphs.py"): run_command("generate_final_graphs.py", s)
                if os.path.exists("generate_academic_report.py"): run_command("generate_academic_report.py", s)
                if os.path.exists("generate_word_report.py"): run_command("generate_word_report.py", s)
                s.update(label="Analisi OK", state="complete")
        
        st.success("✅ Pipeline Terminata!")

# --- TAB 2: RISULTATI ---
with tab2:
    if os.path.exists(RESULTS_CSV):
        df = pd.read_csv(RESULTS_CSV)
        if 'mode' in df.columns: df = df.drop(columns=['mode'])
        
        # Evidenziamo il miglior modello
        if not df.empty:
            best = df.loc[df['f1_macro'].idxmax()]
            c1, c2 = st.columns(2)
            c1.metric("Miglior F1-Score", f"{best['f1_macro']:.4f}")
            c2.metric("Top Modello", best['model'])
            st.dataframe(df.style.background_gradient(subset=['f1_macro'], cmap="Greens"), use_container_width=True)
        
        # Galleria Grafici
        cols = st.columns(2)
        imgs = ["Figure_2_ROC_PR_Comparison.png", "Figure_3_Confusion_Matrix_XGBoost.png", "Figure_5_Performance_Tradeoff.png"]
        for i, img in enumerate(imgs):
            if os.path.exists(f"reports/{img}"):
                cols[i%2].image(f"reports/{img}", caption=img)
    else:
        st.info("Esegui la pipeline per vedere i risultati.")

# --- TAB 3: DOWNLOAD ---
with tab3:
    st.header("📥 Download Artefatti")
    c1, c2, c3 = st.columns(3)
    with c1:
        if os.path.exists("TESI_DOTTORATO_COMPLETA.docx"):
            with open("TESI_DOTTORATO_COMPLETA.docx", "rb") as f: st.download_button("📘 Tesi (.docx)", f, "Tesi.docx")
        if os.path.exists("PHD_FINAL_DISSERTATION.md"):
            with open("PHD_FINAL_DISSERTATION.md", "r") as f: st.download_button("📝 Report (.md)", f, "Report.md")
    with c2:
        if os.path.exists(RESULTS_CSV):
            with open(RESULTS_CSV, "r") as f: st.download_button("💾 Risultati CSV", f.read(), "results.csv")
    with c3:
        if os.path.exists(MODEL_FILE):
            with open(MODEL_FILE, "rb") as f: st.download_button("📦 Modello XGBoost", f, "xgboost.pkl")

# --- TAB 4: INVESTIGATORE (CON LOGICA N/D) ---
with tab4:
    st.header("🕵️ Investigatore Forense")
    
    files_ok = os.path.exists(MODEL_FILE) and os.path.exists(FEATURES_CONFIG)
    
    if files_ok:
        try:
            le = joblib.load(ENCODER_FILE)
            model = joblib.load(MODEL_FILE)
            with open(FEATURES_CONFIG, 'r') as f: feat_list = json.load(f)
            
            # Parsing Feature per Menu
            # Includiamo T (Tecniche) e TA (Tattiche)
            ttp_cols = [c for c in feat_list if (c.startswith("T") and c[1].isdigit()) or (c.startswith("TA") and c[2].isdigit())]
            country_cols = [c for c in feat_list if "country" in c.lower()]
            sector_cols = [c for c in feat_list if "sector" in c.lower()]
            
            country_map = {c.replace("victim_country_", "").replace("country_", ""): c for c in country_cols}
            sector_map = {c.replace("victim_sector_", "").replace("sector_", ""): c for c in sector_cols}
            
            # --- SIMULATORE DATI REALI ---
            st.markdown("### 🧪 Simulazione Casi Reali")
            if os.path.exists(os.path.join(DATA_DIR, "y_val.csv")):
                y_val = pd.read_csv(os.path.join(DATA_DIR, "y_val.csv"))
                X_val = pd.read_csv(os.path.join(DATA_DIR, "X_val.csv"))
                top_gangs = y_val['label_gang'].value_counts().head(20).index.tolist()
                target_gang = st.selectbox("Carica Profilo Gang:", ["Seleziona..."] + top_gangs)
                
                default_ttps = []
                default_country_idx, default_sector_idx = 0, 0
                
                if target_gang != "Seleziona...":
                    # Pesca un esempio reale a caso
                    idx = np.random.choice(y_val[y_val['label_gang'] == target_gang].index)
                    row = X_val.loc[idx]
                    active = row[row == 1].index.tolist()
                    
                    # Filtro di sicurezza per evitare crash su feature non presenti nel menu
                    default_ttps = [c for c in active if c in ttp_cols]
                    
                    # Logica estrazione paese/settore
                    c_found = [c for c in active if "country" in c]
                    if c_found: 
                        clean = c_found[0].replace("victim_country_", "").replace("country_", "")
                        if clean in country_map: default_country_idx = sorted(list(country_map.keys())).index(clean) + 1
                    
                    s_found = [c for c in active if "sector" in c]
                    if s_found: 
                        clean = s_found[0].replace("victim_sector_", "").replace("sector_", "")
                        if clean in sector_map: default_sector_idx = sorted(list(sector_map.keys())).index(clean) + 1
                    
                    st.success(f"✅ Profilo **{target_gang}** caricato! (Attivate {len(default_ttps)} feature)")
            
            st.divider()

            # --- INPUT UTENTE ---
            c1, c2 = st.columns([2, 1])
            with c1: 
                sel_ttps = st.multiselect("Tecniche & Tattiche (TTPs):", ttp_cols, default=default_ttps)
            with c2: 
                sel_country = st.selectbox("Paese:", ["Sconosciuto"] + sorted(list(country_map.keys())), index=default_country_idx)
                sel_sector = st.selectbox("Settore:", ["Sconosciuto"] + sorted(list(sector_map.keys())), index=default_sector_idx)
            
            if st.button("🔍 IDENTIFICA GANG", type="primary", use_container_width=True):
                # Costruzione Vettore
                vec = np.zeros((1, len(feat_list)))
                active_count = 0
                
                for t in sel_ttps: 
                    vec[0, feat_list.index(t)] = 1
                    active_count += 1
                if sel_country != "Sconosciuto": 
                    vec[0, feat_list.index(country_map[sel_country])] = 1
                    active_count += 1
                if sel_sector != "Sconosciuto": 
                    vec[0, feat_list.index(sector_map[sel_sector])] = 1
                    active_count += 1
                
                if active_count == 0:
                    st.error("Seleziona almeno un dato per l'analisi.")
                else:
                    # Predizione
                    probs = model.predict_proba(pd.DataFrame(vec, columns=feat_list))[0]
                    best_idx = np.argmax(probs)
                    raw_gang = le.inverse_transform([best_idx])[0]
                    conf = probs[best_idx] * 100
                    
                    # === LOGICA "N/D" (SOGLIA DI SICUREZZA) ===
                    THRESHOLD = 50.0  # Sotto il 50% consideriamo il risultato inaffidabile
                    
                    st.divider()
                    
                    if conf < THRESHOLD:
                        st.warning(f"⚠️ **Risultato: N/D (Non Disponibile)**")
                        st.markdown(f"Il modello ha rilevato una compatibilità con **{raw_gang}**, ma la confidenza ({conf:.2f}%) è insufficiente per un'attribuzione forense certa.")
                        st.info("💡 **Suggerimento:** Aggiungi più TTPs, o specifica il Settore/Paese della vittima per ridurre l'ambiguità.")
                    else:
                        st.balloons() if conf > 85 else None
                        color = "green" if conf > 75 else "orange"
                        st.markdown(f"### Gang Identificata: :{color}[{raw_gang}]")
                        st.metric("Confidenza Attribuzione", f"{conf:.2f}%")
                        if conf > 90: st.caption("Attribuzione di Alto Livello (High Confidence)")
                    
                    with st.expander("📊 Dettagli Probabilità (Top 5)"):
                        top5 = np.argsort(probs)[::-1][:5]
                        for i in top5:
                            g = le.inverse_transform([i])[0]
                            st.write(f"- **{g}**: {probs[i]*100:.2f}%")

        except Exception as e: st.error(f"Errore caricamento: {e}. Fai RESET FILES.")
    else: st.warning("Modello non trovato. Esegui il 'Training Modelli' nel Tab 1!")