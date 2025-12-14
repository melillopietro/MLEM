import requests
import json
import os

def download_mitre_data():
    print("🌐 Connecting to MITRE CTI Repository...")
    # URL ufficiale del framework Enterprise ATT&CK in formato JSON
    url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            mitre_dict = {}
            
            print(f"📦 Processing {len(data['objects'])} STIX objects...")
            
            for obj in data['objects']:
                # Filtriamo solo le "attack-pattern" (le Tecniche)
                if obj.get('type') == 'attack-pattern':
                    # Cerchiamo l'ID esterno (es. T1059)
                    external_refs = obj.get('external_references', [])
                    mitre_id = None
                    for ref in external_refs:
                        if ref.get('source_name') == 'mitre-attack':
                            mitre_id = ref.get('external_id')
                            break
                    
                    if mitre_id:
                        # Salviamo ID: Nome e Descrizione (prima frase)
                        desc = obj.get('description', 'No description available.')
                        # Pulizia base della descrizione (prendiamo solo la prima frase per brevità)
                        short_desc = desc.split('. ')[0] + "."
                        mitre_dict[mitre_id] = {
                            "name": obj.get('name'),
                            "description": short_desc
                        }
            
            # Salviamo su file
            output_file = "mitre_definitions.json"
            with open(output_file, "w") as f:
                json.dump(mitre_dict, f, indent=4)
            
            print(f"✅ Success! Database saved to '{output_file}' with {len(mitre_dict)} techniques.")
        else:
            print("❌ Failed to download data.")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    download_mitre_data()