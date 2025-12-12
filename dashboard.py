import streamlit as st
import pandas as pd
import os
import subprocess
import sys
import time
import requests

# === CONFIGURAZIONE ===
st.set_page_config(page_title="MLEM Dashboard", page_icon="🛡️", layout="wide")

RAW_DATASET_XLSX = "Dataset Ransomware.xlsx"
RAW_DATASET_CSV = "Dataset Normalized.csv"
RESULTS_CSV = "model_comparison_results_final.csv"

st.title("🛡️ MLEM: Ransomware Attribution Pipeline")

# === SIDEBAR ===
with st.sidebar:
    st.header("🎛️ Pannello di Controllo")
    
    # 1. INPUT
    st.markdown("### 1. Dati")
    source = st.radio("Dataset:", ["Default", "Upload Locale"], label_visibility="collapsed")
    if source == "Upload Locale":
        up = st.file_uploader("Carica Excel/CSV", type=['xlsx', 'csv'])
        if up and st.button("Usa questo file"):
            dest = RAW_DATASET_XLSX if up.name.endswith('.xlsx') else RAW_DATASET_CSV
            with open(dest, "wb") as f: f.write(up.getbuffer())
            st.success("✅ Caricato!")

    st.divider()

    # 2. PARAMETRI SIMULAZIONE (NUOVO!)
    st.markdown("### 2. 🛠️ Parametri Simulazione")
    st.info("Modifica questi valori per testare diverse configurazioni.")
    
    # Slider per i parametri
    n_estimators = st.slider("Numero Alberi (Estimators)", min_value=10, max_value=500, value=150, step=10, help="Più alto = più preciso ma più lento.")
    max_depth = st.slider("Profondità Max (Depth)", min_value=3, max_value=50, value=10, step=1, help="Controlla la complessità del modello.")
    n_iter = st.slider("Iterazioni Search", min_value=1, max_value=20, value=5, help="Quanti tentativi fare per gli altri parametri.")
    
    st.divider()

    # 3. AZIONI
    st.markdown("### 3. Fasi Pipeline")
    run_prep = st.checkbox("Preprocessing", value=False)
    run_train = st.checkbox("Training Modelli", value=True)
    run_anal = st.checkbox("Analisi & Report", value=True)

# === FUNZIONE ESECUZIONE ===
def run_command(cmd_args, log_box):
    # Se ricevo una lista, la uso. Se stringa, la incapsulo.
    if isinstance(cmd_args, str):
        cmd = [sys.executable, cmd_args]
    else:
        cmd = [sys.executable] + cmd_args

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
            text=True, encoding='utf-8', errors='replace', env=env
        )
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None: break
            if line:
                log_box.code(line.strip(), language="bash")
                time.sleep(0.001)

        if process.returncode == 0: return True
        else:
            log_box.error(f"Errore: {cmd}")
            st.error(process.stderr.read())
            return False
    except Exception as e:
        log_box.error(f"Eccezione: {e}")
        return False

# === TAB PRINCIPALI ===
tab1, tab2, tab3 = st.tabs(["🚀 Esecuzione", "📊 Risultati", "📥 Download"])

# --- TAB 1: ESECUZIONE ---
with tab1:
    if st.button("▶️ AVVIA SIMULAZIONE", type="primary", use_container_width=True):
        
        if run_prep:
            with st.status("Preprocessing...", expanded=False) as s:
                run_command("normalized_dataset.py", s)
                run_command("dataset_ML_Formatter.py", s)
                run_command("generate_dataset.py", s)
                run_command("stratification_dataset.py", s)
                s.update(label="Preprocessing OK", state="complete")

        if run_train:
            with st.status(f"Training (Est: {n_estimators}, Depth: {max_depth})...", expanded=True) as s:
                
                # COSTRUZIONE COMANDO CON PARAMETRI
                # Passiamo i valori degli slider allo script python
                cmd_train = [
                    "training_manager.py",
                    "--n_estimators", str(n_estimators),
                    "--max_depth", str(max_depth),
                    "--n_iter", str(n_iter)
                ]
                
                if run_command(cmd_train, s):
                    s.write("ML Training OK")
                
                if os.path.exists("NN_new.py"):
                    try: 
                        import tensorflow
                        run_command("NN_new.py", s)
                    except: pass
                
                if os.path.exists("add_NN_tocompare.py"):
                    run_command("add_NN_tocompare.py", s)
                
                s.update(label="Training Completato", state="complete")

        if run_anal:
            with st.status("Analisi e Report...", expanded=False) as s:
                if os.path.exists("final_fixed.py"): run_command("final_fixed.py", s)
                if os.path.exists("generate_final_graphs.py"): run_command("generate_final_graphs.py", s)
                if os.path.exists("generate_academic_report.py"): run_command("generate_academic_report.py", s)
                if os.path.exists("generate_word_report.py"): run_command("generate_word_report.py", s)
                s.update(label="Analisi OK", state="complete")

        st.success("✅ Simulazione Terminata!")

# --- TAB 2: RISULTATI ---
with tab2:
    if os.path.exists(RESULTS_CSV):
        df = pd.read_csv(RESULTS_CSV)
        # Pulizia colonne extra
        if 'mode' in df.columns: df = df.drop(columns=['mode'])
        
        best = df.loc[df['f1_macro'].idxmax()]
        
        c1, c2 = st.columns(2)
        c1.metric("Miglior F1", f"{best['f1_macro']:.4f}")
        c2.metric("Top Modello", best['model'])
        
        st.dataframe(df.style.background_gradient(subset=['f1_macro'], cmap="Greens"))
        
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
        st.subheader("📄 Documenti")
        if os.path.exists("TESI_DOTTORATO_COMPLETA.docx"):
            with open("TESI_DOTTORATO_COMPLETA.docx", "rb") as f:
                st.download_button("📘 Scarica Tesi (.docx)", f, "Tesi.docx")
        if os.path.exists("PHD_FINAL_DISSERTATION.md"):
            with open("PHD_FINAL_DISSERTATION.md", "r", encoding="utf-8") as f:
                st.download_button("📝 Scarica Report (.md)", f.read(), "Report.md")

    with c2:
        st.subheader("📊 Dati")
        if os.path.exists(RESULTS_CSV):
            with open(RESULTS_CSV, "r") as f:
                st.download_button("💾 Tabella Risultati", f.read(), "results.csv")
    
    with c3:
        st.subheader("🧠 Modelli")
        if os.path.exists("XGBoost_best_model.pkl"):
            with open("XGBoost_best_model.pkl", "rb") as f:
                st.download_button("📦 Scarica XGBoost", f, "xgboost.pkl")