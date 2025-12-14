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

# === SAFETY LAYER 2: Conditional Import for PLOTLY (Maps/Heatmaps) ===
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# === CONFIGURATION ===
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

    # 2. CYBER THREAT GLOBE (3D INTERACTIVE)
    st.subheader("Global Threat Intelligence Center")
    st.markdown("Real-time visualization of Ransomware Victimology density.")
    
    if PLOTLY_AVAILABLE:
        try:
            map_source_file = os.path.join(DATA_DIR, "X_train.csv")
            if os.path.exists(map_source_file):
                X_map = pd.read_csv(map_source_file)
                country_cols = [c for c in X_map.columns if "country" in c]
                
                if country_cols:
                    if len(X_map) > 5000: X_map_sample = X_map.sample(5000, random_state=42)
                    else: X_map_sample = X_map

                    active_countries = []
                    for col in country_cols:
                        count = X_map_sample[col].sum()
                        if count > 0:
                            clean_name = col.replace("victim_country_", "").replace("country_", "")
                            active_countries.extend([clean_name] * int(count))
                    
                    map_df = pd.DataFrame(active_countries, columns=['Nation'])
                    map_counts = map_df['Nation'].value_counts().reset_index()
                    map_counts.columns = ['Nation', 'Attacks']

                    if not map_counts.empty:
                        fig_map = px.choropleth(
                            map_counts, locations="Nation", locationmode='country names',
                            color="Attacks", hover_name="Nation",
                            color_continuous_scale="Reds", title="<b>LIVE ATTACK DENSITY</b>",
                            projection="orthographic"
                        )
                        fig_map.update_layout(
                            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                            geo=dict(showframe=False, showcoastlines=True, coastlinecolor="RebeccaPurple",
                                projection_type='orthographic', showocean=True, oceancolor="rgb(10, 10, 20)",
                                showlakes=True, lakecolor="rgb(10, 10, 20)", landcolor="rgb(30, 30, 40)", bgcolor="rgba(0,0,0,0)"),
                            margin={"r":0,"t":50,"l":0,"b":0}, height=600
                        )
                        col_map, col_list = st.columns([3, 1])
                        with col_map: st.plotly_chart(fig_map, use_container_width=True)
                        with col_list:
                            st.markdown("####  Top Targets")
                            st.dataframe(map_counts.head(10).style.background_gradient(cmap="Reds"), hide_index=True, use_container_width=True)
                    else: st.warning("No geographic data found.")
                else: st.warning("Dataset missing 'country' columns.")
        except Exception as e: st.error(f"Visualization Error: {e}")
    else: st.warning("Plotly not installed.")

    st.divider()

    # 3. ADVANCED TACTICAL ANALYSIS (HEATMAP)
    st.subheader("Tactical Overlap Analysis (TTP Heatmap)")
    st.markdown("Visual correlation between Ransomware Families and MITRE ATT&CK Techniques.")
    
    if PLOTLY_AVAILABLE:
        try:
            if os.path.exists(os.path.join(DATA_DIR, "X_train.csv")) and os.path.exists(os.path.join(DATA_DIR, "y_train.csv")):
                X_h = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
                y_h = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
                
                df_heat = X_h.copy()
                df_heat['Gang'] = y_h['label_gang']
                top_gangs = df_heat['Gang'].value_counts().head(10).index
                df_heat_top = df_heat[df_heat['Gang'].isin(top_gangs)]
                ttp_cols_heat = [c for c in df_heat.columns if (c.startswith("T") and c[1].isdigit())]
                
                heatmap_data = df_heat_top.groupby('Gang')[ttp_cols_heat].mean()
                heatmap_data = heatmap_data.loc[:, (heatmap_data > 0.1).any(axis=0)]
                
                if not heatmap_data.empty:
                    fig_heat = px.imshow(
                        heatmap_data, labels=dict(x="MITRE Technique", y="Ransomware Family", color="Usage Frequency"),
                        x=heatmap_data.columns, y=heatmap_data.index,
                        color_continuous_scale="Viridis", aspect="auto"
                    )
                    fig_heat.update_layout(title="<b>Signature Fingerprinting: Who uses what?</b>", height=500)
                    st.plotly_chart(fig_heat, use_container_width=True)
                    st.caption("Insight: Darker areas show distinct techniques. Vertical bands show shared tools (Affiliate Overlap).")
                else: st.warning("Not enough TTP density.")
        except Exception as e: st.warning(f"Heatmap Error: {e}")

    st.divider()

    # 4. THREAT ACTOR ENCYCLOPEDIA
    st.subheader("Threat Actor Profiling System")
    st.markdown("Automated generation of behavioral profiles based on historical data.")
    
    if os.path.exists(os.path.join(DATA_DIR, "y_train.csv")):
        y_prof = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
        all_gangs = sorted(y_prof['label_gang'].unique())
        
        col_sel, col_stats = st.columns([1, 3])
        with col_sel: selected_gang = st.selectbox("Select Threat Actor:", all_gangs)
        
        with col_stats:
            if selected_gang and os.path.exists(os.path.join(DATA_DIR, "X_train.csv")):
                indices = y_prof[y_prof['label_gang'] == selected_gang].index
                X_prof = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv")).iloc[indices]
                
                sec_cols = [c for c in X_prof.columns if "sector" in c]
                top_sectors = X_prof[sec_cols].sum().sort_values(ascending=False).head(3)
                
                cnt_cols = [c for c in X_prof.columns if "country" in c]
                top_countries = X_prof[cnt_cols].sum().sort_values(ascending=False).head(3)
                
                tech_cols = [c for c in X_prof.columns if c.startswith("T") and c[1].isdigit()]
                top_techs = X_prof[tech_cols].sum().sort_values(ascending=False).head(5)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("** Preferred Sectors**")
                    if not top_sectors.empty and top_sectors.max() > 0:
                        for idx, val in top_sectors.items():
                            name = idx.replace("victim_sector_", "").replace("sector_", "")
                            st.progress(int(val/len(X_prof)*100), text=f"{name}")
                    else: st.write("No distinct sector pattern.")

                with c2:
                    st.markdown("** Preferred Targets**")
                    if not top_countries.empty and top_countries.max() > 0:
                        for idx, val in top_countries.items():
                            name = idx.replace("victim_country_", "").replace("country_", "")
                            st.write(f"📍 **{name}**")
                    else: st.write("Global/Random targeting.")

                with c3:
                    st.markdown("**🛠️ Key TTPs (Modus Operandi)**")
                    for idx, val in top_techs.items(): st.code(idx, language="text")
    st.divider()

    # 5. SCIENTIFIC CLUSTERING (PCA Visualization)
    st.subheader("Gang Similarity Clusters (PCA Projection)")
    st.markdown("2D projection of the high-dimensional feature space to visualize logical distance between groups.")
    
    if PLOTLY_AVAILABLE:
        try:
            from sklearn.decomposition import PCA
            
            if os.path.exists(os.path.join(DATA_DIR, "X_train.csv")) and os.path.exists(os.path.join(DATA_DIR, "y_train.csv")):
                X_pca = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
                y_pca = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
                
                df_pca = X_pca.copy()
                df_pca['Label'] = y_pca['label_gang']
                if len(df_pca) > 1000: df_pca = df_pca.sample(1000, random_state=42)
                
                features_only = df_pca.drop(columns=['Label'])
                pca = PCA(n_components=2)
                components = pca.fit_transform(features_only)
                
                fig_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
                fig_df['Gang'] = df_pca['Label'].values
                
                fig_cluster = px.scatter(
                    fig_df, x='PC1', y='PC2', color='Gang',
                    title="<b>Semantic Similarity Space</b> (Closer points = Similar TTPs/Targets)",
                    opacity=0.7, hover_data=['Gang'],
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                fig_cluster.update_layout(
                    template="plotly_dark", height=500,
                    xaxis_title="Principal Component 1", yaxis_title="Principal Component 2"
                )
                st.plotly_chart(fig_cluster, use_container_width=True)
                st.caption("Insight: Distinct clusters indicate distinct modus operandi. Overlapping clusters imply strong tactical similarities.")
        except Exception as e: st.warning(f"Clustering Error: {e}")

    st.divider()

    # 6. ATTACK SOPHISTICATION METRICS
    st.subheader("Operational Sophistication Analysis")
    st.markdown("Quantifying the complexity of attacks based on the number of TTPs employed per incident.")
    
    if PLOTLY_AVAILABLE:
        try:
            if os.path.exists(os.path.join(DATA_DIR, "X_train.csv")) and os.path.exists(os.path.join(DATA_DIR, "y_train.csv")):
                X_s = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
                y_s = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
                
                tech_cols_only = [c for c in X_s.columns if (c.startswith("T") and c[1].isdigit())]
                df_soph = pd.DataFrame()
                df_soph['Gang'] = y_s['label_gang']
                df_soph['Complexity'] = X_s[tech_cols_only].sum(axis=1)
                
                soph_ranking = df_soph.groupby('Gang')['Complexity'].mean().sort_values(ascending=False).head(15)
                
                fig_soph = px.bar(
                    soph_ranking, orientation='h',
                    x=soph_ranking.values, y=soph_ranking.index,
                    title="<b>Average Attack Chain Complexity (Top 15 Groups)</b>",
                    labels={'x': 'Avg. Unique Techniques per Attack', 'y': 'Ransomware Family'},
                    color=soph_ranking.values, color_continuous_scale="Plasma"
                )
                fig_soph.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_soph, use_container_width=True)
                st.caption("Insight: Higher complexity often indicates advanced APT-like capabilities (e.g., Conti, LockBit).")
        except Exception as e: st.warning(f"Sophistication Analysis Error: {e}")

    st.divider()

    # 7. GLOBAL FEATURE IMPORTANCE
    st.subheader("Global Explainability (Top Discriminative Features)")
    
    if PLOTLY_AVAILABLE and os.path.exists(MODEL_FILE) and os.path.exists(FEATURES_CONFIG):
        try:
            model_glob = joblib.load(MODEL_FILE)
            with open(FEATURES_CONFIG, 'r') as f: f_list = json.load(f)
            if hasattr(model_glob, "feature_importances_"):
                imp_df = pd.DataFrame({'Feature': f_list, 'Importance': model_glob.feature_importances_}).sort_values(by='Importance', ascending=False).head(10)
                fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title="Top 10 Influential Features (XGBoost)", color='Importance', color_continuous_scale='Viridis')
                fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as e: st.warning(f"Feature Importance Error: {e}")

    st.divider()

    # 8. STATIC REPORTS
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
                
                if active_count == 0: st.error("Please select at least one feature.")
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
                                    fig, ax = plt.subplots(figsize=(10, 5))
                                    shap.plots.waterfall(shap_val_class, max_display=10, show=False)
                                    st.pyplot(fig)
                                except Exception as e: st.warning(f"SHAP Analysis unavailable: {e}")

                    with st.expander("View Full Probability Distribution"):
                        top5 = np.argsort(probs)[::-1][:5]
                        for i in top5:
                            g = le.inverse_transform([i])[0]
                            st.write(f"- **{g}**: {probs[i]*100:.2f}%")

        except Exception as e: st.error(f"System Error: {e}. Please reset the cache.")
    else: st.warning("Model not found. Please execute the Training Pipeline in Tab 1.")