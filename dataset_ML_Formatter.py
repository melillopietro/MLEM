import pandas as pd
import re

# === CONFIGURAZIONE ===
standardized_file = "Dataset Normalized.csv"  # File CSV normalizzato (Victim Country, Victim sectors, ecc.)
original_file = "Dataset Ransomware.xlsx"     # Contiene il foglio "Ransomware Gang Profile"
output_file = "final_ml_dataset_encoded.csv"

# === LETTURA DEI FILE ===
df_std = pd.read_csv(standardized_file)
df_profile = pd.read_excel(original_file, sheet_name="Ransomware Gang Profile")

# === CONVERSIONE FORMATO DATA DELL'ATTACCO ===
df_std["date"] = pd.to_datetime(df_std["date"], errors="coerce")

# === UNIONE TTP (rimozione dei NaN) ===
ttp_cols = ['TTPS'] + [f'TTPS.{i}' for i in range(1, 111) if f'TTPS.{i}' in df_profile.columns]
df_profile['All_TTPs'] = df_profile[ttp_cols].apply(
    lambda row: ','.join(row.dropna().astype(str)), axis=1
)

# === PROFILO GANG (solo colonne utili) ===
df_profile_reduced = df_profile[['Gang name', 'All_TTPs']]

# === MERGE VITTIME + GANG PROFILE ===
df_merged = df_std.merge(df_profile_reduced, left_on='gang', right_on='Gang name', how='left')

# === FILTRA LE RIGHE SENZA TTP ===
df_merged = df_merged[df_merged['All_TTPs'].notna() & (df_merged['All_TTPs'].str.strip() != '')]

# === COSTRUZIONE DATASET PER ML ===
df_final = df_merged[[
    'gang',
    'All_TTPs',
    'date',
    'Victim sectors',
    'Victim Country'
]].rename(columns={
    'gang': 'label_gang',
    'All_TTPs': 'gang_ttps',
    'date': 'attack_date',
    'Victim sectors': 'victim_sector',
    'Victim Country': 'victim_country'
})

def clean_ttps(ttps_string):
    if pd.isna(ttps_string):
        return ''
    # Split e rimuovi caratteri invisibili da ogni TTP
    ttps = str(ttps_string).split(',')
    cleaned_ttps = []
    for t in ttps:
        # Rimuove spazi, caratteri Unicode invisibili, BOM, ecc.
        t_clean = re.sub(r'[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]', '', t)
        t_clean = re.sub(r'\s+', '', t_clean.strip().upper())  # spazi e uppercase
        if t_clean:
            cleaned_ttps.append(t_clean)
    return ','.join(sorted(set(cleaned_ttps)))


df_final['gang_ttps'] = df_final['gang_ttps'].apply(clean_ttps)

# === ESTRAZIONE FEATURE TEMPORALI ===
df_final["attack_date"] = pd.to_datetime(df_final["attack_date"], errors="coerce")
df_final["year"] = df_final["attack_date"].dt.year
df_final["month"] = df_final["attack_date"].dt.month
df_final["dayofweek"] = df_final["attack_date"].dt.dayofweek

# === ONE-HOT ENCODING: Settori e Paesi ===
df_encoded = pd.get_dummies(df_final, columns=["victim_sector", "victim_country"])

# === ONE-HOT ENCODING: TTP ===
df_ttp = df_final['gang_ttps'].str.get_dummies(sep=',')

# === CONCATENAZIONE FINALE ===
df_encoded = pd.concat([
    df_encoded.drop(columns=['gang_ttps', 'attack_date']),
    df_ttp
], axis=1)

# === CONVERSIONE BOOLEANI IN 0/1 ===
bool_cols = df_encoded.select_dtypes(include='bool').columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

# === SALVATAGGIO CSV ===
df_encoded.to_csv(output_file, index=False)
print(f"✔ Dataset ML salvato in formato CSV: {output_file}")
