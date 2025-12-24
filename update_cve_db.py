import json
import requests
import os
import sys

# === CONFIGURAZIONE FONTI UFFICIALI ===
CISA_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
MITRE_DB_FILE = "mitre_definitions.json"
OUTPUT_FILE = "cve_definitions.json"


def download_cisa_feed():
    """Scarica il catalogo ufficiale CISA delle vulnerabilità sfruttate."""
    print(f"📡 Connecting to CISA Government Feed ({CISA_KEV_URL})...")
    try:
        response = requests.get(CISA_KEV_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        print(f"✅ Download Successful: Retrieved {data['count']} active vulnerabilities.")
        return data['vulnerabilities']
    except Exception as e:
        print(f"❌ Error downloading CISA Feed: {e}")
        return []


def load_mitre_keywords():
    """Carica le TTP MITRE per creare un dizionario di mapping inverso."""
    if not os.path.exists(MITRE_DB_FILE):
        print("⚠️ MITRE DB not found. Skipping intelligent mapping.")
        return {}

    with open(MITRE_DB_FILE, 'r') as f:
        mitre_data = json.load(f)

    # Creiamo un dizionario {Keyword: TTP_ID}
    # Es: {"phishing": "T1566", "powershell": "T1059"}
    keyword_map = {}
    for tid, content in mitre_data.items():
        name = content['name'].lower()
        keyword_map[name] = tid
        # Aggiungiamo anche parole chiave dalla descrizione (opzionale, semplificato qui)

    print(f"🧠 Loaded {len(keyword_map)} MITRE heuristics for mapping.")
    return keyword_map


def enrich_and_map(vulnerabilities, mitre_map):
    """
    Motore di Intelligenza: Collega CVE -> TTP analizzando il testo.
    """
    print("⚙️ Running Heuristic Mapping Engine (CVE <-> MITRE)...")

    processed_db = {}

    for vuln in vulnerabilities:
        cve_id = vuln['cveID']
        desc = vuln['shortDescription'].lower()
        name = vuln['vulnerabilityName']

        # 1. Trova TTP basandosi sulle parole chiave nella descrizione
        found_ttps = []
        if mitre_map:
            for keyword, tid in mitre_map.items():
                # Euristica semplice: se la tecnica è menzionata nella descrizione della CVE
                if keyword in desc or keyword in name.lower():
                    found_ttps.append(tid)

        # Fallback: Se non trova nulla, mappa su vettori comuni (Exploit Public Facing)
        if not found_ttps:
            if "execute code" in desc or "execution" in desc: found_ttps.append("T1203")  # Exploit for Client Execution
            if "escalation" in desc or "privilege" in desc: found_ttps.append(
                "T1068")  # Exploitation for Privilege Escalation

        # Deduplica
        found_ttps = list(set(found_ttps))

        # Costruzione Record Finale
        processed_db[cve_id] = {
            "name": name,
            "severity": "Active Exploitation (CISA KEV)",  # CISA non da lo score numerico nel JSON light, ma lo stato
            "description": vuln['shortDescription'],
            "linked_ttps": found_ttps,
            "target_sector": "All/Unknown",  # CISA non specifica il settore
            "vendor": vuln['vendorProject'],
            "product": vuln['product'],
            "date_added": vuln['dateAdded']
        }

    return processed_db


def main():
    # 1. Scarica CVE Reali
    cve_list = download_cisa_feed()
    if not cve_list: return

    # 2. Carica intelligenza MITRE
    mitre_map = load_mitre_keywords()

    # 3. Collega i puntini
    final_db = enrich_and_map(cve_list, mitre_map)

    # 4. Salva
    with open(OUTPUT_FILE, "w") as f:
        json.dump(final_db, f, indent=4)

    print(f"🎉 Success! Generated production-grade DB with {len(final_db)} linked entries.")
    print(f"📂 Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()