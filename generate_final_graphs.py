import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.preprocessing import LabelEncoder, label_binarize
from itertools import cycle

DATA_DIR = "./dataset_split"
OUTPUT_DIR = "./reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("📂 Caricamento Dati...")
try:
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))
    cols = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"), nrows=1).columns.tolist()
    X_test = X_test[cols]
except: exit()

le = LabelEncoder()
y_test_enc = le.fit_transform(y_test["label_gang"])
y_test_bin = label_binarize(y_test_enc, classes=range(len(le.classes_)))

models_to_plot = {"XGBoost": "XGBoost_best_model.pkl", "RandomForest": "RandomForest_best_model.pkl"}

# 1. ROC
print("📊 ROC...")
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
colors = cycle(['blue', 'green'])
stats = {}
for (name, fname), color in zip(models_to_plot.items(), colors):
    if os.path.exists(fname):
        try:
            pipeline = joblib.load(fname)
            if hasattr(pipeline, "predict_proba"):
                y_score = pipeline.predict_proba(X_test)
                fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
                plt.plot(fpr, tpr, color=color, lw=2, label=f'{name}')
                precision, recall, _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
                stats[name] = {'pr': (recall, precision), 'color': color}
        except: pass
plt.plot([0, 1], [0, 1], 'k--'); plt.legend(); plt.title("ROC")

plt.subplot(1, 2, 2)
for name, data in stats.items():
    plt.plot(data['pr'][0], data['pr'][1], color=data['color'], lw=2, label=f'{name}')
plt.legend(); plt.title("PR Curve")
plt.savefig(os.path.join(OUTPUT_DIR, "Figure_2_ROC_PR_Comparison.png"), dpi=300)

# 2. CM & Features
if os.path.exists("XGBoost_best_model.pkl"):
    print("📊 Matrix & Features...")
    pipeline = joblib.load("XGBoost_best_model.pkl")
    cm = confusion_matrix(y_test_enc, pipeline.predict(X_test))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    plt.figure(figsize=(10, 8)); sns.heatmap(cm_norm, cmap='Blues', vmax=0.8); plt.savefig(os.path.join(OUTPUT_DIR, "Figure_3_Confusion_Matrix_XGBoost.png"), dpi=300)
    
    model = pipeline.named_steps['classifier'] if hasattr(pipeline, "named_steps") else pipeline
    if hasattr(model, "feature_importances_"):
        idx = np.argsort(model.feature_importances_)[::-1][:20]
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(idx)), model.feature_importances_[idx], align="center", color='teal')
        plt.yticks(range(len(idx)), [cols[i] for i in idx]); plt.gca().invert_yaxis()
        plt.title("Feature Importance"); plt.savefig(os.path.join(OUTPUT_DIR, "Figure_4_Feature_Importance.png"), dpi=300)

print("✅ Finito.")