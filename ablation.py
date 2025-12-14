import pandas as pd
import plotly.express as px
import os

# === CONFIGURAZIONE ===
INPUT_FILE = "dataset_split/X_train.csv"
OUTPUT_FILE = "mappa_tesi.html"

print("Generazione Mappa Vittimologia per la Slide 9...")

try:
    if not os.path.exists(INPUT_FILE):
        print(f"Errore: {INPUT_FILE} non trovato.")
        exit()

    # Carica i dati
    df = pd.read_csv(INPUT_FILE)
    
    # Cerca le colonne relative ai paesi (quelle che iniziano con "victim_country_" o "country_")
    country_cols = [c for c in df.columns if "country" in c]
    
    if not country_cols:
        print("Nessuna colonna geografica trovata nel dataset.")
        exit()

    print(f"Trovate {len(country_cols)} nazioni. Elaborazione...")

    # Funzione per estrarre il nome del paese dalla riga (one-hot decoding)
    # Prende un campione di 5000 righe per non appesantire troppo se il dataset è enorme
    if len(df) > 5000:
        df = df.sample(5000, random_state=42)

    def get_active_country(row):
        for col in country_cols:
            if row[col] == 1:
                # Pulisce il nome (toglie i prefissi)
                clean_name = col.replace("victim_country_", "").replace("country_", "")
                return clean_name
        return None

    # Applica la funzione
    df['Nation'] = df.apply(get_active_country, axis=1)

    # Conta gli attacchi per nazione
    counts = df['Nation'].value_counts().reset_index()
    counts.columns = ['Nation', 'Attacks']
    
    # Rimuove valori nulli o "Unknown"
    counts = counts[counts['Nation'].notna()]
    counts = counts[counts['Nation'] != 'Unknown']

    # Crea la mappa 3D/Mondo
    fig = px.choropleth(
        counts,
        locations="Nation",
        locationmode='country names',
        color="Attacks",
        hover_name="Nation",
        color_continuous_scale=px.colors.sequential.Reds,
        title="MLEM - Global Ransomware Victimology Distribution",
        projection="natural earth" # Oppure 'orthographic' per il mappamondo 3D tondo
    )

    fig.update_layout(
        margin={"r":0,"t":50,"l":0,"b":0},
        title_font_size=24
    )

    # Salva
    fig.write_html(OUTPUT_FILE)
    print(f"Fatto! Apri il file '{OUTPUT_FILE}' nel browser e fai lo screenshot.")

except Exception as e:
    print(f"Errore: {e}")