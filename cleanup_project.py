import os
import shutil

# === CONFIGURAZIONE ===
ARCHIVE_DIR = "archive"

# LISTA COMPLETA DEI FILE OBSOLETI DA SPOSTARE
FILES_TO_MOVE = [
    # 1. Vecchi Entry Points
    "main.py",
    "main_orchestrator.py",
    "app_orchestrator.py",           # Sostituito da dashboard.py
    "app_orchestrator_final.py",     # Se hai rinominato in dashboard.py, questo è un duplicato
    
    # 2. Vecchie Pipeline di Training
    "multimodel_gridsearch_pipeline.py", # Il vecchio script lento
    "resume_pipeline_fixed.py",          # Script temporaneo
    "resume_pipeline_fixed_win.py",      # Script temporaneo
    
    # 3. Script di Analisi Superati
    "compare_saved_models.py",       # Ora i risultati sono nella dashboard
    "generate_automatic_report.py",  # Sostituito da generate_academic_report
    
    # 4. Versioni vecchie dei generatori grafici
    "generate_report_plots.py",
    "generate_report_plots_v2.py",
    "generate_report_plots_v3.py",
    
    # 5. File CSV vecchi/parziali (tieni solo il _final)
    "model_comparison_results.csv",  # Quello vecchio (il nuovo ha _final)
    "label_mapping.csv"              # Spesso ricreato dinamicamente
]

def cleanup():
    print("🧹 Inizio pulizia approfondita del progetto...")
    
    # Crea cartella archive se non esiste
    if not os.path.exists(ARCHIVE_DIR):
        os.makedirs(ARCHIVE_DIR)
        print(f"📂 Creata cartella '{ARCHIVE_DIR}'")

    moved_count = 0
    
    for filename in FILES_TO_MOVE:
        if os.path.exists(filename):
            try:
                destination = os.path.join(ARCHIVE_DIR, filename)
                
                # Se il file esiste già nell'archivio, lo sovrascriviamo
                if os.path.exists(destination):
                    os.remove(destination)
                    
                shutil.move(filename, destination)
                print(f"✅ Archiviato: {filename}")
                moved_count += 1
            except Exception as e:
                print(f"❌ Errore su {filename}: {e}")
    
    print("\n" + "="*40)
    print(f"✨ PULIZIA COMPLETATA! Spostati {moved_count} file in '{ARCHIVE_DIR}/'.")
    print("   La tua root ora contiene solo la Dashboard e i file Core.")
    print("="*40)

if __name__ == "__main__":
    cleanup()