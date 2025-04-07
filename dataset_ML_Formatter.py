import pandas as pd
import re
'''
Author: Pietro Melillo
Date: 2025-04-07
Description: script that creates a dataset for ML from the CSV file containing the TTPs of the ransomware gangs.
            The dataset is created by merging the TTPs with the dataset of victims, which contains information about the sectors and countries of the victims.
            The dataset is then preprocessed and one-hot encoded to create a dataset suitable for ML.
            '''


standardized_file = "Dataset Normalized.csv"  # CSV file created by the script "dataset_normalizer.py"
original_file = "Dataset Ransomware.xlsx"     # File with the TTPs of the gangs (tab "Ransomware Gang Profile")
output_file = "final_ml_dataset_encoded.csv"  # Output file for the ML dataset

df_std = pd.read_csv(standardized_file)
df_profile = pd.read_excel(original_file, sheet_name="Ransomware Gang Profile")

df_std["date"] = pd.to_datetime(df_std["date"], errors="coerce")

ttp_cols = ['TTPS'] + [f'TTPS.{i}' for i in range(1, 111) if f'TTPS.{i}' in df_profile.columns]
df_profile['All_TTPs'] = df_profile[ttp_cols].apply(
    lambda row: ','.join(row.dropna().astype(str)), axis=1
)

df_profile_reduced = df_profile[['Gang name', 'All_TTPs']]

df_merged = df_std.merge(df_profile_reduced, left_on='gang', right_on='Gang name', how='left')

df_merged = df_merged[df_merged['All_TTPs'].notna() & (df_merged['All_TTPs'].str.strip() != '')]

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

# Preprocessing: deletes spaces and other characters from the TTPs
def clean_ttps(ttps_string):
    if pd.isna(ttps_string):
        return ''
    ttps = str(ttps_string).split(',')
    cleaned_ttps = []
    for t in ttps:
        t_clean = re.sub(r'[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]', '', t)
        t_clean = re.sub(r'\s+', '', t_clean.strip().upper())  # spazi e uppercase
        if t_clean:
            cleaned_ttps.append(t_clean)
    return ','.join(sorted(set(cleaned_ttps)))


df_final['gang_ttps'] = df_final['gang_ttps'].apply(clean_ttps)

df_final["attack_date"] = pd.to_datetime(df_final["attack_date"], errors="coerce")
df_final["year"] = df_final["attack_date"].dt.year
df_final["month"] = df_final["attack_date"].dt.month
df_final["dayofweek"] = df_final["attack_date"].dt.dayofweek

# One-hot encoding for categorical variables: creates a new column for each unique value in the categorical columns
df_encoded = pd.get_dummies(df_final, columns=["victim_sector", "victim_country"])

# One-hot encoding for TTPs: creates a new column for each unique TTP
df_ttp = df_final['gang_ttps'].str.get_dummies(sep=',')


df_encoded = pd.concat([
    df_encoded.drop(columns=['gang_ttps', 'attack_date']),
    df_ttp
], axis=1)

# Convert boolean columns to int (0/1)
bool_cols = df_encoded.select_dtypes(include='bool').columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

df_encoded.to_csv(output_file, index=False)
print(f"Dataset saved : {output_file}")
