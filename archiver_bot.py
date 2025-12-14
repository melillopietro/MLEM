import os
import shutil
import time
from datetime import datetime

# === CONFIGURAZIONE ===
SOURCE_DIR = "."              # Cartella corrente
ARCHIVE_DIR = "./archive"     # Dove spostare i file
DAYS_OLD = 30                 # Sposta file non modificati da X giorni
# Estensioni da spostare A PRESCINDERE dalla data (es. file temporanei, log, zip vecchi)
TRASH_EXTENSIONS = ['.tmp', '.bak', '.log', '.old'] 
# Vuoi spostare anche i vecchi grafici/report? Aggiungi: '.png', '.html', '.txt'

# Metti False se vuoi spostare davvero i file. True fa solo finta (stampa a video)
DRY_RUN = True  

def move_files():
    # Crea cartella archive se non esiste
    if not os.path.exists(ARCHIVE_DIR):
        os.makedirs(ARCHIVE_DIR)
        print(f"📁 Creata cartella archivio: {ARCHIVE_DIR}")

    count = 0
    now = time.time()
    cutoff = now - (DAYS_OLD * 86400) # 86400 secondi in un giorno

    print(f"🔍 Scansione in corso... Cerco file più vecchi di {DAYS_OLD} giorni o con estensioni: {TRASH_EXTENSIONS}")
    print("-" * 60)

    for root, dirs, files in os.walk(SOURCE_DIR):
        # Evita di scansionare la cartella archive stessa o cartelle nascoste (.git)
        if "archive" in root or ".git" in root or "__pycache__" in root:
            continue

        for filename in files:
            filepath = os.path.join(root, filename)
            file_ext = os.path.splitext(filename)[1].lower()

            try:
                # Ottieni data ultima modifica
                file_mtime = os.path.getmtime(filepath)
                
                # CRITERIO 1: È un'estensione "spazzatura"?
                is_junk_ext = file_ext in TRASH_EXTENSIONS
                
                # CRITERIO 2: È vecchio?
                is_old = file_mtime < cutoff

                # CRITERIO SPECIALE: Non spostare lo script stesso o file essenziali
                if filename in ["archiver_bot.py", "requirements.txt", ".gitignore", "README.md"]:
                    continue

                if is_old or is_junk_ext:
                    reason = "VECCHIO" if is_old else "ESTENSIONE"
                    date_str = datetime.fromtimestamp(file_mtime).strftime('%Y-%m-%d')
                    
                    if DRY_RUN:
                        print(f"[SIMULAZIONE] Sposterei: {filename} ({reason} - {date_str})")
                    else:
                        # Gestione nome duplicato in archive
                        dest_path = os.path.join(ARCHIVE_DIR, filename)
                        if os.path.exists(dest_path):
                            base, ext = os.path.splitext(filename)
                            timestamp = int(time.time())
                            dest_path = os.path.join(ARCHIVE_DIR, f"{base}_{timestamp}{ext}")
                        
                        shutil.move(filepath, dest_path)
                        print(f"✅ Spostato: {filename} -> archive/")
                    
                    count += 1

            except Exception as e:
                print(f"❌ Errore su {filename}: {e}")

    print("-" * 60)
    if DRY_RUN:
        print(f"🏁 SIMULAZIONE COMPLETATA. Trovati {count} file candidati.")
        print("💡 Imposta DRY_RUN = False nello script per eseguire lo spostamento reale.")
    else:
        print(f"🏁 OPERAZIONE COMPLETATA. Spostati {count} file in '{ARCHIVE_DIR}'.")

if __name__ == "__main__":
    move_files()