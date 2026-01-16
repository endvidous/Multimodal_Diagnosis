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
import sys
import argparse
from pathlib import Path
from difflib import SequenceMatcher

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.symptom_normalizer import normalize_symptom, SYNONYM_MAP, TYPO_MAP

# Paths
symptom_path = project_root / "data" / "symptom_vocabulary.json"
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
    # Normalize the input symptom first
    normalized = normalize_symptom(external, apply_synonyms=False)
    
    # Check if typo was fixed
    if external.lower() != normalized:
        print(f"  [NORMALIZED] '{external}' -> '{normalized}'")
    
    # Check for synonym mapping suggestion
    if external.lower() in SYNONYM_MAP:
        canonical = SYNONYM_MAP[external.lower()]
        print(f"  [SYNONYM] Consider using canonical form: '{canonical}'")
    
    # Check if normalized form is an exact match
    if normalized in [s.lower() for s in ALL_SYMPTOMS]:
        for s in ALL_SYMPTOMS:
            if s.lower() == normalized:
                print(f"  '{external}' -> '{s}' (exact match)")
                return s
    
    # Also check original (in case normalization was too aggressive)
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


def process_disease(disease: str, info: dict, auto: bool = False) -> list:
    """Process all symptoms for a disease."""
    mayo_symptoms = info.get('mayo_clinic_symptoms', [])
    
    if not mayo_symptoms:
        if not auto:
            print(f"\n  [EMPTY] No symptoms provided yet")
        return []
    
    if not auto:
        print(f"\n  Processing {len(mayo_symptoms)} symptoms...")
        
    mapped = []
    
    for sym in mayo_symptoms:
        if auto:
            # Auto-mode: Only accept exact matches (post-normalization)
            normalized = normalize_symptom(sym, apply_synonyms=True)
            if normalized in [s.lower() for s in ALL_SYMPTOMS]:
                # Find correct casing
                for s in ALL_SYMPTOMS:
                    if s.lower() == normalized:
                        mapped.append(s)
                        break
        else:
            # Interactive mode
            result = map_symptom_interactive(sym)
            if result:
                mapped.append(result)
    
    return list(set(mapped))  # Remove duplicates


def main():
    """Main interactive mapping workflow."""
    parser = argparse.ArgumentParser(description="Map symptoms to vocabulary")
    parser.add_argument("--auto", action="store_true", help="Auto-accept exact matches only (no prompts)")
    parser.add_argument("--remap", action="store_true", help="Reprocess ALL diseases (even if already mapped)")
    parser.add_argument("--list", action="store_true", help="List all vocabulary symptoms")
    args = parser.parse_args()

    if args.list:
        list_vocabulary()
        return

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
    if output_path.exists() and not args.remap:
        with open(output_path) as f:
            mapped_data = json.load(f)
        print(f"Loaded {len(mapped_data)} existing mappings")
    else:
        mapped_data = {}
        if args.remap:
            print("Force-remapping ALL diseases...")
    
    # Find diseases to process
    to_process = []
    for disease, info in template.items():
        mayo_symptoms = info.get('mayo_clinic_symptoms', [])
        
        # Skip if empty
        if not mayo_symptoms:
            continue
            
        # If auto/remap or not yet mapped, process it
        if args.remap or disease not in mapped_data:
            to_process.append((disease, info))
    
    print(f"\nDiseases to process: {len(to_process)}")
    
    if not to_process:
        print("Nothing to do.")
        return
    
    if args.auto:
        print("\nStarting AUTO mapping (exact matches only)...")
    else:
        print("\nStarting INTERACTIVE mapping...")
        print("Commands: Enter number to select, 's' to skip, 'q' to quit and save")
    
    print("-"*70)
    
    count_updated = 0
    
    for i, (disease, info) in enumerate(to_process, 1):
        if not args.auto:
            print(f"\n[{i}/{len(to_process)}] {disease.upper()}")
            print(f"  Current samples: {info.get('current_samples', '?')}")
            
            action = input("  Process this disease? (y/n/q): ").strip().lower()
            if action == 'q':
                break
            elif action != 'y':
                continue
        
        mapped_symptoms = process_disease(disease, info, auto=args.auto)
        
        if mapped_symptoms:
            # Only save if we actually found something
            mapped_data[disease] = {
                "mapped_symptoms": mapped_symptoms,
                "original_count": info.get('current_samples', 0)
            }
            count_updated += 1
            if not args.auto:
               print(f"\n  Mapped {len(mapped_symptoms)} symptoms for {disease}")
            
            # Save incrementally
            with open(output_path, 'w') as f:
                json.dump(mapped_data, f, indent=2)
        elif args.auto:
             # In auto mode, if we found nothing, maybe keep old mapping if exists? 
             # For now, let's assume we want to refresh.
             pass
                
    # Final save
    with open(output_path, 'w') as f:
        json.dump(mapped_data, f, indent=2)
    
    print("\n" + "="*70)
    print(f"COMPLETE: Updated {count_updated} diseases")
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
    
    main()
