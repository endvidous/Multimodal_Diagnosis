"""
Generate Synthetic Data: Creates training samples from mapped symptoms.

WORKFLOW:
1. Run symptom_mapper.py first to create mapped symptoms
2. Run this script to generate synthetic training data
3. Output: data/processed/symptoms/symptoms_to_disease_augmented.csv
"""

import pandas as pd
import numpy as np
import json
import random
from pathlib import Path

# Paths
project_root = Path(__file__).parent.parent
data_path = project_root / "data" / "processed" / "symptoms" / "symptoms_to_disease_cleaned.csv"
output_path = project_root / "data" / "processed" / "symptoms" / "symptoms_to_disease_augmented.csv"
symptom_path = project_root / "models" / "checkpoints" / "symptom_columns.json"
mapping_path = project_root / "data" / "disease_mapping.json"
mapped_symptoms_path = project_root / "data" / "rare_diseases_symptoms_mapped.json"

# Load data
print("Loading data...")
df = pd.read_csv(data_path)

with open(symptom_path) as f:
    ALL_SYMPTOMS = json.load(f)

with open(mapping_path) as f:
    category_map = json.load(f)

# Create disease -> category mapping
disease_to_category = {}
for cat, diseases in category_map.items():
    for d in diseases:
        disease_to_category[d] = cat

print(f"Original dataset: {len(df):,} rows, {df['diseases'].nunique()} diseases")


def generate_synthetic_samples(disease: str, symptoms: list, n_samples: int = 25,
                                min_symptoms: int = 4, max_symptoms: int = 8) -> list:
    """Generate synthetic training samples for a disease."""
    samples = []
    category = disease_to_category.get(disease, "Unknown Type")
    
    for _ in range(n_samples):
        # Random subset of symptoms
        n_sym = random.randint(min_symptoms, min(max_symptoms, len(symptoms)))
        selected = random.sample(symptoms, n_sym)
        
        # Create row with all symptoms as 0/1
        row = {col: 0 for col in ALL_SYMPTOMS}
        for sym in selected:
            if sym in row:
                row[sym] = 1
        
        row['diseases'] = disease
        row['disease_category'] = category
        row['symptoms'] = ", ".join(selected)
        
        samples.append(row)
    
    return samples


def main():
    """Generate synthetic data from mapped symptoms."""
    print("\n" + "="*70)
    print("SYNTHETIC DATA GENERATOR")
    print("="*70)
    
    # Load mapped symptoms
    if not mapped_symptoms_path.exists():
        print(f"\nNo mapped symptoms found: {mapped_symptoms_path}")
        print("Run: python scripts/symptom_mapper.py")
        return
    
    with open(mapped_symptoms_path) as f:
        mapped_data = json.load(f)
    
    print(f"Found {len(mapped_data)} diseases with mapped symptoms")
    
    # Get current counts
    counts = df['diseases'].value_counts()
    
    # Generate synthetic samples
    random.seed(42)
    all_synthetic = []
    target_samples = 25  # Minimum samples per disease
    
    for disease, info in mapped_data.items():
        symptoms = info.get('mapped_symptoms', [])
        current_count = counts.get(disease, 0)
        
        if not symptoms:
            print(f"  [SKIP] {disease}: no symptoms")
            continue
        
        if len(symptoms) < 4:
            print(f"  [WARN] {disease}: only {len(symptoms)} symptoms (need 4+)")
            continue
        
        if current_count >= target_samples:
            print(f"  [OK] {disease}: already has {current_count} samples")
            continue
        
        n_new = target_samples - current_count
        samples = generate_synthetic_samples(disease, symptoms, n_samples=n_new)
        all_synthetic.extend(samples)
        print(f"  [ADD] {disease}: {current_count} -> {current_count + n_new} samples (+{n_new})")
    
    if not all_synthetic:
        print("\nNo synthetic samples generated.")
        return
    
    # Create DataFrame
    df_synthetic = pd.DataFrame(all_synthetic)
    
    # Reorder columns to match original
    cols = df.columns.tolist()
    df_synthetic = df_synthetic[cols]
    
    # Combine with original
    df_combined = pd.concat([df, df_synthetic], ignore_index=True)
    
    print(f"\n{'='*70}")
    print(f"SUMMARY:")
    print(f"  Original samples: {len(df):,}")
    print(f"  Synthetic samples: {len(df_synthetic):,}")
    print(f"  Total samples: {len(df_combined):,}")
    print(f"{'='*70}")
    
    # Ask for confirmation
    confirm = input("\nSave augmented dataset? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    # Save
    df_combined.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    
    # Show new counts
    print(f"\nUpdated disease counts:")
    new_counts = df_combined['diseases'].value_counts()
    for disease in mapped_data.keys():
        old = counts.get(disease, 0)
        new = new_counts.get(disease, 0)
        if old < 25:
            print(f"  {disease}: {old} -> {new}")


if __name__ == "__main__":
    main()
