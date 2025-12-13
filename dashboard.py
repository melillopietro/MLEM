import streamlit as st
import pandas as pd
import os
import subprocess
import sys
import time
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt

# === SAFETY LAYER 1: Conditional Import for SHAP (XAI) ===
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# === SAFETY LAYER 2: Conditional Import for PLOTLY (Maps) ===
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# === CONFIGURATION ===
# Layout wide, formal title, standard favicon
st.set_page_config(page_title="MLEM Framework", layout="wide")

RESULTS_CSV = "model_comparison_results_final.csv"
FEATURES_CONFIG = "features_config.json"
MODEL_FILE = "XGBoost_best_model.pkl"
ENCODER_FILE = "label_encoder.pkl"
DATA_DIR = "./dataset_split"
RAW_DATASET_XLSX = "Dataset Ransomware.xlsx"
RAW_DATASET_CSV = "Dataset Normalized.csv"

st.title("MLEM: Ransomware Attribution Framework")
st.markdown("**Advanced Hybrid Profiling & Forensic Attribution System**")

# === UTILS ===
def run_command(cmd_args, log_box):
    if isinstance(cmd_args, str): cmd = [sys.executable, cmd_args]
    else: cmd = [sys.executable] + cmd_args 
    env = os.environ.copy(); env["PYTHONIOENCODING"] = "utf-8"
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', env=env)
        while True:
            line = p.stdout.readline()
            if not line and p.poll() is not None: break
            if line: log_box.code(line.strip(), language="bash"); time.sleep(0.001)
        return p.returncode == 0
    except Exception as e: log_box.error(f"Error: {e}"); return False

# === SIDEBAR ===
with st.sidebar:
    st.header("Control Panel")
    
    st.markdown("### 1. Data Source")
    source = st.radio("Source:", ["Default Dataset", "Local Upload"], label_visibility="collapsed")
    if source == "Local Upload":
        up = st.file_uploader("Upload Excel/CSV", type=['xlsx', 'csv'])
        if up and st.button("Load File"):
            dest = RAW_DATASET_XLSX if up.name.endswith('.xlsx') else RAW_DATASET_CSV
            with open(dest, "wb") as f: f.write(up.getbuffer())
            st.success("File uploaded successfully.")
            
    st.divider()
    st.markdown("### 2. Hyperparameters")
    n_estimators = st.slider("Number of Estimators (Trees)", 50, 500, 150, step=10)
    max_depth = st.slider("Max Tree Depth", 3, 50, 15, step=1)
    
    st.divider()
    st.markdown("### 3. Pipeline Stages")
    run_prep = st.checkbox("1. Data Preprocessing", value=False)
    run_train = st.checkbox("2. Model Training", value=True)
    run_anal = st.checkbox("3. Analysis & Reporting", value=True)
    
    st.divider()
    if st.button("RESET SYSTEM CACHE"):
        for f in [RESULTS_CSV, MODEL_FILE, ENCODER_FILE, FEATURES_CONFIG, "RandomForest_best_model.pkl"]:
            if os.path.exists(f): os.remove(f)
        st.warning("Cache cleared. Please restart the pipeline.")

# === TABS ===
tab1, tab2, tab3, tab4 = st.tabs(["Pipeline Execution", "Results & Intelligence", "Downloads", "Forensic Investigator"])

# --- TAB 1: EXECUTION ---
with tab1:
    st.subheader("Automated Machine Learning Pipeline")
    if st.button("RUN FULL PIPELINE", type="primary", use_container_width=True):
        if run_prep:
            with st.status("Preprocessing Data...", expanded=False) as s:
                run_command("normalized_dataset.py", s); run_command("dataset_ML_Formatter.py", s)
                run_command("generate_dataset.py", s); run_command("stratification_dataset.py", s)
                s.update(label="Preprocessing Completed", state="complete")

        if run_train:
            with st.status(f"Training Models (Trees: {n_estimators}, Depth: {max_depth})...", expanded=True) as s:
                cmd = ["training_manager.py", "--n_estimators", str(n_estimators), "--max_depth", str(max_depth)]
                run_command(cmd, s)
                # Run Neural Networks if available
                if os.path.exists("NN_new.py"): 
                    try: import tensorflow; run_command("NN_new.py", s)
                    except: pass
                s.update(label="Training Completed", state="complete")

        if run_anal:
            with st.status("Generating Technical Reports...", expanded=False) as s:
                if os.path.exists("final_fixed.py"): run_command("final_fixed.py", s)
                if os.path.exists("generate_final_graphs.py"): run_command("generate_final_graphs.py", s)
                s.update(label="Analysis Completed", state="complete")
        st.success("Pipeline executed successfully.")

# --- TAB 2: RESULTS ---
with tab2:
    st.header("Performance Metrics & Global Intelligence")
    
    # 1. LEADERBOARD
    if os.path.exists(RESULTS_CSV):
        df = pd.read_csv(RESULTS_CSV)
        if 'mode' in df.columns: df = df.drop(columns=['mode'])
        if not df.empty:
            best = df.loc[df['f1_macro'].idxmax()]
            c1, c2, c3 = st.columns(3)
            c1.metric("Best F1-Score (Macro)", f"{best['f1_macro']:.4f}")
            c2.metric("Top Accuracy", f"{best['accuracy']*100:.2f}%")
            c3.metric("Best Performing Model", best['model'])
            st.dataframe(df.style.background_gradient(subset=['f1_macro'], cmap="Greens"), use_container_width=True)
    else:
        st.info("Run the pipeline to view results.")

    st.divider()

    # 2. CYBER THREAT MAP
    st.subheader("Global Victimology Map (Geospatial Analysis)")
    
    if PLOTLY_AVAILABLE:
        try:
            if os.path.exists(os.path.join(DATA_DIR, "X_val.csv")):
                X_val_map = pd.read_csv(os.path.join(DATA_DIR, "X_val.csv"))
                country_cols = [c for c in X_val_map.columns if "country" in c]
                
                if country_cols:
                    def get_country(row):
                        for c in country_cols:
                            if row[c] == 1: return c.replace("victim_country_", "").replace("country_", "")
                        return None

                    map_data = X_val_map.copy()
                    if len(map_data) > 2000: map_data = map_data.sample(2000)
                    
                    map_data['Nation'] = map_data.apply(get_country, axis=1)
                    counts = map_data['Nation'].value_counts().reset_index()
                    counts.columns = ['Nation', 'Attacks']
                    counts = counts[counts['Nation'].notna()]

                    if not counts.empty:
                        fig_map = px.choropleth(
                            counts, locations="Nation", locationmode='country names',
                            color="Attacks", hover_name="Nation",
                            color_continuous_scale=px.colors.sequential.Reds,
                            title="Global Distribution of Ransomware Victims (Validation Set)"
                        )
                        st.plotly_chart(fig_map, use_container_width=True)
                    else: st.warning("No geographic data found for visualization.")
        except Exception as e: st.warning(f"Map rendering error: {e}")
    else:
        st.warning("Plotly library not installed. Map visualization disabled.")

    st.divider()

    # 3. GLOBAL FEATURE IMPORTANCE
    st.subheader("Global Explainability (Top Discriminative Features)")
    st.caption("Which features are most critical for the model to distinguish between Ransomware groups?")
    
    if PLOTLY_AVAILABLE:
        if os.path.exists(MODEL_FILE) and os.path.exists(FEATURES_CONFIG):
            try:
                model_glob = joblib.load(MODEL_FILE)
                with open(FEATURES_CONFIG, 'r') as f: f_list = json.load(f)
                
                if hasattr(model_glob, "feature_importances_"):
                    imp_df = pd.DataFrame({
                        'Feature': f_list,
                        'Importance': model_glob.feature_importances_
                    }).sort_values(by='Importance', ascending=False).head(10)
                    
                    fig_imp = px.bar(
                        imp_df, x='Importance', y='Feature', orientation='h',
                        title="Top 10 Influential Features (XGBoost)",
                        color='Importance', color_continuous_scale='Viridis'
                    )
                    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_imp, use_container_width=True)
                else:
                    st.info("The current model does not support native feature importance.")
            except Exception as e: st.warning(f"Feature Importance Error: {e}")
    else:
        st.info("Install Plotly to view the feature importance chart.")

    st.divider()

    # 4. STATIC REPORTS
    st.subheader("Statistical Reports")
    cols = st.columns(2)
    imgs = ["Figure_2_ROC_PR_Comparison.png", "Figure_3_Confusion_Matrix_XGBoost.png"]
    for i, img in enumerate(imgs):
        if os.path.exists(f"reports/{img}"): cols[i%2].image(f"reports/{img}", caption=img)

# --- TAB 3: DOWNLOADS ---
with tab3:
    st.header("Data & Artifacts Export")
    c1, c2, c3 = st.columns(3)
    with c1:
        if os.path.exists("TESI_DOTTORATO_COMPLETA.docx"):
            with open("TESI_DOTTORATO_COMPLETA.docx", "rb") as f: st.download_button("Download Thesis (.docx)", f, "Thesis.docx")
    with c2:
        if os.path.exists(RESULTS_CSV):
            with open(RESULTS_CSV, "r") as f: st.download_button("Download Results (.csv)", f.read(), "results.csv")
    with c3:
        if os.path.exists(MODEL_FILE):
            with open(MODEL_FILE, "rb") as f: st.download_button("Download Model (.pkl)", f, "xgboost_model.pkl")

# --- TAB 4: INVESTIGATOR ---
with tab4:
    st.header("Forensic Investigator & Local XAI")
    
    if os.path.exists(MODEL_FILE) and os.path.exists(FEATURES_CONFIG):
        try:
            le = joblib.load(ENCODER_FILE)
            model = joblib.load(MODEL_FILE)
            with open(FEATURES_CONFIG, 'r') as f: feat_list = json.load(f)
            
            ttp_cols = [c for c in feat_list if (c.startswith("T") and c[1].isdigit()) or (c.startswith("TA") and c[2].isdigit())]
            country_cols = [c for c in feat_list if "country" in c.lower()]
            sector_cols = [c for c in feat_list if "sector" in c.lower()]
            
            country_map = {c.replace("victim_country_", "").replace("country_", ""): c for c in country_cols}
            sector_map = {c.replace("victim_sector_", "").replace("sector_", ""): c for c in sector_cols}
            
            # SIMULATOR
            st.markdown("### Real-World Scenario Simulator")
            if os.path.exists(os.path.join(DATA_DIR, "y_val.csv")):
                y_val = pd.read_csv(os.path.join(DATA_DIR, "y_val.csv"))
                X_val = pd.read_csv(os.path.join(DATA_DIR, "X_val.csv"))
                top_gangs = y_val['label_gang'].value_counts().head(20).index.tolist()
                target_gang = st.selectbox("Load Threat Actor Profile:", ["Select..."] + top_gangs)
                
                default_ttps = []
                default_country_idx, default_sector_idx = 0, 0
                
                if target_gang != "Select...":
                    idx = np.random.choice(y_val[y_val['label_gang'] == target_gang].index)
                    row = X_val.loc[idx]
                    active = row[row == 1].index.tolist()
                    default_ttps = [c for c in active if c in ttp_cols]
                    
                    c_found = [c for c in active if "country" in c]
                    if c_found: 
                        clean = c_found[0].replace("victim_country_", "").replace("country_", "")
                        if clean in country_map: default_country_idx = sorted(list(country_map.keys())).index(clean) + 1
                    
                    s_found = [c for c in active if "sector" in c]
                    if s_found: 
                        clean = s_found[0].replace("victim_sector_", "").replace("sector_", "")
                        if clean in sector_map: default_sector_idx = sorted(list(sector_map.keys())).index(clean) + 1
                    
                    st.success(f"Profile **{target_gang}** loaded successfully (Sample ID: {idx})")
            
            st.divider()

            # INPUT FORM
            c1, c2 = st.columns([2, 1])
            with c1: sel_ttps = st.multiselect("Observed TTPs (Techniques):", ttp_cols, default=default_ttps)
            with c2: 
                sel_country = st.selectbox("Victim Country:", ["Unknown"] + sorted(list(country_map.keys())), index=default_country_idx)
                sel_sector = st.selectbox("Victim Sector:", ["Unknown"] + sorted(list(sector_map.keys())), index=default_sector_idx)
            
            if st.button("IDENTIFY THREAT ACTOR", type="primary", use_container_width=True):
                vec = np.zeros((1, len(feat_list)))
                active_count = 0
                for t in sel_ttps: vec[0, feat_list.index(t)] = 1; active_count += 1
                if sel_country != "Unknown": vec[0, feat_list.index(country_map[sel_country])] = 1; active_count += 1
                if sel_sector != "Unknown": vec[0, feat_list.index(sector_map[sel_sector])] = 1; active_count += 1
                
                if active_count == 0:
                    st.error("Please select at least one feature.")
                else:
                    input_df = pd.DataFrame(vec, columns=feat_list)
                    probs = model.predict_proba(input_df)[0]
                    best_idx = np.argmax(probs)
                    raw_gang = le.inverse_transform([best_idx])[0]
                    conf = probs[best_idx] * 100
                    
                    st.divider()
                    
                    if conf < 50.0:
                        st.warning(f"**Result: N/A (Inconclusive Analysis)**")
                        st.markdown(f"Low confidence match for **{raw_gang}** ({conf:.2f}%). Additional evidence required.")
                    else:
                        color = "green" if conf > 75 else "orange"
                        st.markdown(f"### Identified Threat Actor: :{color}[{raw_gang}]")
                        st.metric("Confidence Score", f"{conf:.2f}%")
                        
                        # LOCAL SHAP (Local XAI)
                        is_tree_model = "XGB" in str(type(model)) or "RandomForest" in str(type(model))
                        
                        if SHAP_AVAILABLE and is_tree_model:
                            st.subheader("Explainable AI (Local SHAP Analysis)")
                            st.caption(f"Contribution of each feature to the attribution of {raw_gang}.")
                            
                            with st.spinner("Calculating attribution logic..."):
                                try:
                                    explainer = shap.TreeExplainer(model)
                                    shap_values = explainer(input_df)
                                    if len(shap_values.shape) == 3: shap_val_class = shap_values[0, :, best_idx]
                                    else: shap_val_class = shap_values[0]
                                    
                                    # Create figure
                                    fig, ax = plt.subplots(figsize=(10, 5))
                                    shap.plots.waterfall(shap_val_class, max_display=10, show=False)
                                    st.pyplot(fig)
                                except Exception as e:
                                    st.warning(f"SHAP Analysis unavailable: {e}")

                    with st.expander("View Full Probability Distribution"):
                        top5 = np.argsort(probs)[::-1][:5]
                        for i in top5:
                            g = le.inverse_transform([i])[0]
                            st.write(f"- **{g}**: {probs[i]*100:.2f}%")

        except Exception as e: st.error(f"System Error: {e}. Please reset the cache.")
    else: st.warning("Model not found. Please execute the Training Pipeline in Tab 1.")