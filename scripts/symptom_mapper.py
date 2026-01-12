"""
Symptom Mapper: Maps symptoms from JSON template to the standardized vocabulary.

WORKFLOW:
1. Fill in mayo_clinic_symptoms in data/rare_diseases_symptoms_template.json
2. Run: python scripts/symptom_mapper.py
3. Approve/reject each symptom mapping
4. Output: data/rare_diseases_symptoms_mapped.json (ready for synthetic data generation)
"""

import json
import re
from pathlib import Path
from difflib import SequenceMatcher

# Paths
project_root = Path(__file__).parent.parent
symptom_path = project_root / "models" / "checkpoints" / "symptom_columns.json"
template_path = project_root / "data" / "rare_diseases_symptoms_template.json"
output_path = project_root / "data" / "rare_diseases_symptoms_mapped.json"

# Load symptom vocabulary
with open(symptom_path) as f:
    ALL_SYMPTOMS = json.load(f)
SYMPTOM_SET = set(ALL_SYMPTOMS)

print(f"Loaded {len(ALL_SYMPTOMS)} symptoms from vocabulary")


def tokenize(text: str) -> set:
    """Convert text to lowercase tokens."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return set(text.split())


def similarity_score(a: str, b: str) -> float:
    """Calculate similarity between two strings."""
    tokens_a = tokenize(a)
    tokens_b = tokenize(b)
    
    if not tokens_a or not tokens_b:
        return 0.0
    
    overlap = len(tokens_a & tokens_b)
    total = len(tokens_a | tokens_b)
    jaccard = overlap / total if total > 0 else 0
    
    seq_ratio = SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    return 0.6 * jaccard + 0.4 * seq_ratio


def find_matches(external_symptom: str, top_n: int = 5) -> list:
    """Find best matching symptoms from vocabulary."""
    scores = []
    for symptom in ALL_SYMPTOMS:
        score = similarity_score(external_symptom, symptom)
        scores.append((symptom, score))
    
    scores.sort(key=lambda x: -x[1])
    return scores[:top_n]


def map_symptom_interactive(external: str) -> str | None:
    """Map a single symptom with user confirmation."""
    # Check if already exact match
    if external.lower() in [s.lower() for s in ALL_SYMPTOMS]:
        for s in ALL_SYMPTOMS:
            if s.lower() == external.lower():
                print(f"  '{external}' -> '{s}' (exact match)")
                return s
    
    # Find matches
    matches = find_matches(external)
    
    print(f"\n  '{external}' -> Best matches:")
    for i, (symptom, score) in enumerate(matches, 1):
        pct = int(score * 100)
        print(f"    {i}. [{pct:3d}%] {symptom}")
    
    # Ask for confirmation
    choice = input("  Select (1-5), type exact symptom, or 's' to skip: ").strip()
    
    if choice.lower() == 's':
        return None
    elif choice.isdigit() and 1 <= int(choice) <= 5:
        return matches[int(choice) - 1][0]
    elif choice.lower() in [s.lower() for s in ALL_SYMPTOMS]:
        for s in ALL_SYMPTOMS:
            if s.lower() == choice.lower():
                return s
    else:
        print("  (Invalid choice, skipped)")
        return None


def process_disease(disease: str, info: dict) -> list:
    """Process all symptoms for a disease."""
    mayo_symptoms = info.get('mayo_clinic_symptoms', [])
    
    if not mayo_symptoms:
        print(f"\n  [EMPTY] No symptoms provided yet")
        return []
    
    print(f"\n  Processing {len(mayo_symptoms)} symptoms...")
    mapped = []
    
    for sym in mayo_symptoms:
        result = map_symptom_interactive(sym)
        if result:
            mapped.append(result)
    
    return list(set(mapped))  # Remove duplicates


def main():
    """Main interactive mapping workflow."""
    print("\n" + "="*70)
    print("SYMPTOM MAPPER - Map Mayo Clinic symptoms to vocabulary")
    print("="*70)
    
    # Load template
    if not template_path.exists():
        print(f"\nTemplate not found: {template_path}")
        print("Run: python scripts/create_rare_disease_template.py")
        return
    
    with open(template_path) as f:
        template = json.load(f)
    
    # Load existing mappings if any
    if output_path.exists():
        with open(output_path) as f:
            mapped_data = json.load(f)
        print(f"Loaded {len(mapped_data)} existing mappings")
    else:
        mapped_data = {}
    
    # Find diseases with mayo_clinic_symptoms filled but not yet mapped
    to_process = []
    for disease, info in template.items():
        mayo_symptoms = info.get('mayo_clinic_symptoms', [])
        if mayo_symptoms and disease not in mapped_data:
            to_process.append((disease, info))
    
    print(f"\nDiseases with symptoms to map: {len(to_process)}")
    print(f"Diseases already mapped: {len(mapped_data)}")
    print(f"Diseases still empty: {len(template) - len(to_process) - len(mapped_data)}")
    
    if not to_process:
        print("\nNo new diseases to process.")
        print("Fill in 'mayo_clinic_symptoms' in:")
        print(f"  {template_path}")
        return
    
    print("\nStarting interactive mapping...")
    print("Commands: Enter number to select, 's' to skip, 'q' to quit and save")
    print("-"*70)
    
    for i, (disease, info) in enumerate(to_process, 1):
        print(f"\n[{i}/{len(to_process)}] {disease.upper()}")
        print(f"  Current samples: {info.get('current_samples', '?')}")
        
        action = input("  Process this disease? (y/n/q): ").strip().lower()
        
        if action == 'q':
            break
        elif action != 'y':
            continue
        
        mapped_symptoms = process_disease(disease, info)
        
        if mapped_symptoms:
            mapped_data[disease] = {
                "mapped_symptoms": mapped_symptoms,
                "original_count": info.get('current_samples', 0)
            }
            print(f"\n  Mapped {len(mapped_symptoms)} symptoms for {disease}")
            
            # Save after each disease
            with open(output_path, 'w') as f:
                json.dump(mapped_data, f, indent=2)
    
    # Final save
    with open(output_path, 'w') as f:
        json.dump(mapped_data, f, indent=2)
    
    print("\n" + "="*70)
    print(f"SAVED: {len(mapped_data)} diseases with mapped symptoms")
    print(f"Output: {output_path}")
    print("="*70)


def list_vocabulary():
    """Print all symptoms in vocabulary for reference."""
    print("\n" + "="*70)
    print("ALL 377 SYMPTOMS IN VOCABULARY")
    print("="*70)
    for i, s in enumerate(sorted(ALL_SYMPTOMS), 1):
        print(f"  {i:3d}. {s}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--list':
        list_vocabulary()
    else:
        main()
