"""Fetch complete MITRE ATT&CK Enterprise techniques with descriptions & real-world examples."""

import json
import urllib.request
import ssl
from pathlib import Path

OUTPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "mitre_attack.json"
STIX_URL = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"

def fetch_mitre_data():
    print(f"Downloading official MITRE ATT&CK CTI data from {STIX_URL}...")
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(STIX_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx) as resp:
        stix_data = json.loads(resp.read().decode('utf-8'))
    
    print(f"Loaded {len(stix_data['objects'])} STIX objects. Parsing Enterprise techniques...")

    # Map relationship / group names for examples
    relationships = []
    groups_and_software = {}
    
    techniques = []
    
    for obj in stix_data["objects"]:
        obj_type = obj.get("type")
        if obj_type in ("intrusion-set", "malware", "tool"):
            groups_and_software[obj["id"]] = obj.get("name", "")
        elif obj_type == "relationship" and obj.get("relationship_type") == "uses":
            relationships.append(obj)
            
    # Map technique_id -> list of example usage descriptions / software names
    technique_examples = {}
    for rel in relationships:
        target_ref = rel.get("target_ref")
        source_ref = rel.get("source_ref")
        desc = rel.get("description", "").strip()
        actor = groups_and_software.get(source_ref, "")
        if actor:
            ex_text = f"{actor}: {desc}" if desc else actor
            if target_ref not in technique_examples:
                technique_examples[target_ref] = []
            if len(technique_examples[target_ref]) < 3 and ex_text:
                technique_examples[target_ref].append(ex_text)

    # Process attack patterns (techniques)
    for obj in stix_data["objects"]:
        if obj.get("type") == "attack-pattern":
            # Exclude revoked/deprecated
            if obj.get("revoked") or obj.get("x_mitre_deprecated"):
                continue
            
            # Filter main techniques (exclude sub-techniques)
            if obj.get("x_mitre_is_subtechnique"):
                continue
                
            external_refs = obj.get("external_references", [])
            tech_id = ""
            for ref in external_refs:
                if ref.get("source_name") == "mitre-attack":
                    tech_id = ref.get("external_id", "")
                    break
            
            if not tech_id:
                continue

            tactics = [
                phase.get("phase_name").replace("-", " ").title()
                for phase in obj.get("kill_chain_phases", [])
                if phase.get("kill_chain_name") == "mitre-attack"
            ]

            desc = obj.get("description", "").strip()
            # Truncate long descriptions to first 2-3 sentences for clarity
            sentences = desc.split('. ')
            if len(sentences) > 3:
                short_desc = '. '.join(sentences[:3]) + '.'
            else:
                short_desc = desc

            stix_id = obj.get("id")
            examples_list = technique_examples.get(stix_id, [])
            example_str = " | ".join(examples_list) if examples_list else "No recorded public example."

            techniques.append({
                "id": tech_id,
                "name": obj.get("name", ""),
                "tactics": tactics,
                "description": short_desc,
                "example": example_str
            })

    # Sort by technique ID (e.g., T1001, T1002...)
    techniques.sort(key=lambda x: x["id"])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(techniques, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved {len(techniques)} MITRE ATT&CK techniques to {OUTPUT_FILE}")

if __name__ == "__main__":
    fetch_mitre_data()
