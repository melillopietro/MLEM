import pandas as pd
'''
Author: Pietro Melillo
Date: 2025-04-07
Description: This script normalizes the dataset by mapping sectors to standard categories and normalizing country names.
             It reads an Excel file, processes the data, and saves the normalized dataset as a CSV file.
'''

input_file = "Dataset Ransomware.xlsx"
sheet_name = "Dataset"
output_file = "Dataset Normalized.csv"

sector_mapping = {
    "Healthcare": "Healthcare", "Healtcare": "Healthcare", "Healthcare ": "Healthcare",
    "Healthcare Services": "Healthcare", "Hospitals": "Healthcare", "Hospitals ": "Healthcare",
    "Hospital": "Healthcare", "Hospitals & Physicians Clinics": "Healthcare",
    "Hospitals & Physicians Clinics ": "Healthcare", "Pharmacies": "Healthcare",
    "Elderly Care Services": "Healthcare",
    "Finance": "Finance", "Finance ": "Finance", "Financial": "Finance", "Banking": "Finance",
    "Banking ": "Finance", "Insurance": "Finance", "Credit Cards & Transaction Processing": "Finance",
    "Holding": "Finance", "Holding ": "Finance", "Holding Companies": "Finance",
    "Holding Companies & Conglomerates ": "Finance", "Real Estate": "Finance",
    "Software": "Technology", "IT": "Technology", "Technology": "Technology",
    "Technologies": "Technology", "Sofware": "Technology", "Computer Equipment": "Technology",
    "Networking Software": "Technology", "Internet Service": "Technology",
    "Internet Service ": "Technology", "Internet Service Providers": "Technology",
    "Internet Provider": "Technology", "IoT Solutions": "Technology", "Telecommunications": "Technology",
    "Telecommunication": "Technology", "Electronics": "Technology", "Electronic": "Technology",
    "Business Intelligence": "Technology", "BI": "Technology",
    "Industrial": "Industrial", "Industrial ": "Industrial", "Inustrial": "Industrial",
    "Industrial Machinery & Equipment": "Industrial",
    "Industrial Machinery & Equipment ": "Industrial", "Manufacturing": "Industrial",
    "Manufacturing ": "Industrial", "Manufactoring": "Industrial", "Manufactroring": "Industrial",
    "Aerospace": "Industrial", "Casting": "Industrial", "Chemicals": "Industrial",
    "Chemicals ": "Industrial", "Chemicals & Related Products": "Industrial", "Minerals": "Industrial",
    "Mining": "Industrial", "Mineral Extraction": "Industrial", "Automotive": "Industrial",
    "Automobiles": "Industrial", "Auto": "Industrial", "Auto Industry": "Industrial",
    "Equipment": "Industrial", "Machinery & Equipment": "Industrial",
    "Construction": "Construction", "Construction ": "Construction", "Construcion": "Construction",
    "Costruction": "Construction", "Commercial & Residential Construction": "Construction",
    "Commercial & Residential Construction ": "Construction",
    "Commercial & Residential Constructio": "Construction", "Architecture": "Construction",
    "Architecture, Engineering & Design": "Construction",
    "Architecture, Engineering & Design ": "Construction", "Building Materials": "Construction",
    "Building Materials ": "Construction", "Builing Materials": "Construction",
    "Services": "Services", "Services ": "Services", "Service": "Services",
    "Servoces": "Services", "Business Services": "Services",
    "Business Services ": "Services", "Business association": "Services",
    "Consulting": "Services", "Assurance": "Services", "Support": "Services",
    "Human Resources Software": "Services", "Hospitality": "Services",
    "Accounting Services": "Services", "Accounting": "Services",
    "Hospitality Industry": "Services", "Commercial": "Services",
    "Hospitality & Leisure": "Services", "Hotels & Hospitality": "Services",
    "Retail": "Retail", "retail": "Retail", "Reil": "Retail", "Retial": "Retail",
    "Store": "Retail", "Stores": "Retail",
    "Department Stores, Shopping Centers & Superstores": "Retail",
    "Apparel & Accessories Retail": "Retail", "Appliances": "Retail", "Furniture": "Retail",
    "Household Goods": "Retail", "Husehold Goods": "Retail", "Auctions": "Retail", "Auction": "Retail",
    "Entertainment": "Entertainment", "Gambling": "Entertainment", "Gambling ": "Entertainment",
    "Sport": "Entertainment", "Cultural": "Entertainment", "Attractions": "Entertainment",
    "Media": "Media", "Media ": "Media", "Broadcasting": "Media", "Broadcasting ": "Media",
    "Media & Internet": "Media", "Advertising": "Media",
    "Advertising & Marketing": "Media", "Advertising & Marketing ": "Media",
    "Marketing": "Media", "Design": "Media", "Marshall & Bruce Printing": "Media",
    "Transportation": "Transportation", "Transportation ": "Transportation",
    "Airlines": "Transportation", "Airlanes": "Transportation", "Air Services": "Transportation",
    "Freight & Logistics Services": "Transportation",
    "Freight & Logistics Services ": "Transportation", "Logistics": "Transportation",
    "Logistics ": "Transportation", "Logistics Services": "Transportation",
    "Logistics Services ": "Transportation", "Logistic": "Transportation",
    "Logistic services": "Transportation",
    "Education": "Education", "Education ": "Education", "Education · France": "Education",
    "College": "Education", "Colleges & Universities": "Education",
    "Public": "Public", "Public Administration": "Public", "Federal": "Public", "Fedel": "Public",
    "Local": "Public", "Local ": "Public", "Governative": "Public", "Organization": "Public",
    "Organizations": "Public", "Organizations ": "Public", "Organizat": "Public",
    "No Profit": "Public", "Non-Profit": "Public", "Membership Organizations": "Public",
    "Membership Organizations ": "Public", "Political": "Public", "Politician": "Public",
    "Law": "Legal", "Law ": "Legal",
    "Energy": "Energy", "Eergy": "Energy", "Electricity, Oil & Gas": "Energy",
    "Electricity, Oil & Gas ": "Energy",
    "Agriculture": "Agriculture", "Agricolture": "Agriculture", "Agriculture ": "Agriculture",
    "Food": "Agriculture", "Recycling": "Agriculture",
    "Cosmetics": "Consumer", "Cosemtic": "Consumer", "Consumer": "Consumer",
    "Consumer Services": "Consumer", "Consumer Service": "Consumer", "Household": "Consumer",
    "Unknown": "Unknown", "Not Specified": "Unknown", "nan": "Unknown", "NaN": "Unknown"
}

country_normalization = {
    "usa": "USA", "united states": "USA", "uk": "UK", "uae": "UAE", "india": "India",
    "marocco": "Morocco", "camerun": "Cameroon", "cipro": "Cyprus", "potugal": "Portugal",
    "costarica": "Costa Rica", "republic of vanuatu": "Vanuatu", "republic of palau": "Palau",
    "repubblica di palau": "Palau", "reunion": "Réunion", "lettonia": "Latvia",
    "malvine": "Malvinas", "san salvador": "El Salvador", "scotland": "UK",
    "ukraina": "Ukraine", "south korea": "South Korea", "puerto rico": "Puerto Rico",
    "burkina faso": "Burkina Faso",
    "bosnia and herzegovina and herzegovina": "Bosnia and Herzegovina",
    "bosnia and herzegovina and herzegovina": "Bosnia and Herzegovina",
    "Tunesia": "Tunisia", "tunesia": "Tunisia", "tanzania": "Tanzania",
    "non disponibile": None,
}

def normalize_country(country):
    if pd.isna(country):
        return None
    cleaned = str(country).strip().replace("\xa0", " ").lower()
    return country_normalization.get(cleaned, country.strip())

def map_sector(value):
    if pd.isna(value):
        return "Unknown"
    return sector_mapping.get(str(value).strip(), "Unknown")

df = pd.read_excel(input_file, sheet_name=sheet_name)

df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
df["Victim sectors"] = df["Victim sectors"].apply(map_sector)
df["Victim Country"] = df["Victim Country"].apply(normalize_country)

df.to_csv(output_file, index=False)
print(f"✅ Dataset unificato salvato come: {output_file}")
