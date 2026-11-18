import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.metrics import classification_report

DATA_DIR = "./dataset_split"
OUTPUT_DIR = "./reports"
ENCODER_FILE = "label_encoder.pkl"
MODEL_CANDIDATES = (
    "best_validation_model.pkl",
    "XGBoost_best_model.pkl",
    "RandomForest_best_model.pkl",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def resolve_model_path() -> str | None:
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def fitted_estimator_and_scaler(pipeline):
    if hasattr(pipeline, "named_steps"):
        scaler = pipeline.named_steps.get("scaler")
        if "clf" in pipeline.named_steps:
            return pipeline.named_steps["clf"], scaler
        if "classifier" in pipeline.named_steps:
            return pipeline.named_steps["classifier"], scaler
    return pipeline, None


def is_tree_model(model) -> bool:
    name = model.__class__.__name__
    module = model.__class__.__module__
    blob = f"{module}.{name}".lower()
    return any(
        key in blob
        for key in (
            "xgb",
            "lgbm",
            "forest",
            "gradient",
            "extratrees",
            "histgradient",
        )
    )


# 1. CARICAMENTO DATI
print("📂 Caricamento dati e modello...")
try:
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))
    feature_names = X_train.columns.tolist()
except FileNotFoundError:
    print("❌ Errore: File dataset non trovati. Assicurati di aver eseguito generate_dataset.py")
    raise SystemExit(1)

model_path = resolve_model_path()
if not model_path or not os.path.exists(ENCODER_FILE):
    print("❌ Errore: modello o label_encoder.pkl non trovati. Esegui prima il training.")
    raise SystemExit(1)

pipeline = joblib.load(model_path)
print(f"🔄 Modello caricato: {model_path}")

print("🔄 Allineamento etichette (encoder del training)...")
le = joblib.load(ENCODER_FILE)
y_test_enc = le.transform(y_test["label_gang"])

# 4. GENERAZIONE REPORT DI CLASSIFICAZIONE
print("📊 Generazione Report Dettagliato per Classe...")
try:
    y_pred = pipeline.predict(X_test)
    report = classification_report(
        y_test_enc,
        y_pred,
        labels=range(len(le.classes_)),
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(OUTPUT_DIR, "full_classification_report.csv"))
    print("✅ Report salvato in reports/full_classification_report.csv")
except Exception as e:
    print(f"⚠️ Errore nella generazione del report: {e}")

# 5. SHAP VALUES (Interpretabilità)
print("🧠 Calcolo SHAP Values (potrebbe richiedere 1-2 minuti)...")

try:
    model, scaler = fitted_estimator_and_scaler(pipeline)
    if not is_tree_model(model):
        print(
            f"⚠️ SHAP TreeExplainer non applicato: il modello "
            f"({model.__class__.__name__}) non è un albero."
        )
    else:
        if scaler is not None:
            X_test_scaled = scaler.transform(X_test)
        else:
            X_test_scaled = X_test.values

        X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
        explainer = shap.TreeExplainer(model)
        sample = X_test_df.iloc[:200]
        shap_values = explainer.shap_values(sample, check_additivity=False)

        plt.figure(figsize=(10, 8))
        if isinstance(shap_values, list):
            shap.summary_plot(
                shap_values, sample, feature_names=feature_names, show=False
            )
        else:
            shap.summary_plot(
                shap_values, sample, feature_names=feature_names, show=False
            )
        plt.savefig(
            os.path.join(OUTPUT_DIR, "shap_summary_plot.png"),
            bbox_inches="tight",
            dpi=300,
        )
        print("✅ Grafico SHAP salvato in reports/shap_summary_plot.png")

except Exception as e:
    print(f"⚠️ Errore nel calcolo SHAP: {e}")
    print("   (Nota: Se usi XGBoost > 2.0, assicurati che SHAP sia aggiornato all'ultima versione)")

print("\n🎉 Analisi completata!")