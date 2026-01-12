"""
Symptom Mapping Analysis (READ-ONLY)
Analyzes how well mayo_clinic_symptoms map to the standardized vocabulary.
Does NOT modify any files - just generates a report.
"""

import json
from pathlib import Path
from difflib import SequenceMatcher
import re

# Paths
project_root = Path(__file__).parent.parent
symptom_vocab_path = project_root / "models" / "checkpoints" / "symptom_columns.json"
template_path = project_root / "data" / "rare_diseases_symptoms_template.json"
report_path = project_root / "data" / "symptom_mapping_report.txt"

# Load data
with open(symptom_vocab_path) as f:
    VOCABULARY = json.load(f)
VOCAB_SET = set(s.lower() for s in VOCABULARY)
VOCAB_LOOKUP = {s.lower(): s for s in VOCABULARY}

with open(template_path) as f:
    template = json.load(f)

print(f"Loaded {len(VOCABULARY)} symptoms from vocabulary")
print(f"Loaded {len(template)} diseases from template\n")


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


def find_best_match(symptom: str, top_n: int = 3) -> list:
    """Find best matching symptoms from vocabulary."""
    symptom_lower = symptom.lower().strip()
    
    # Exact match
    if symptom_lower in VOCAB_SET:
        return [(VOCAB_LOOKUP[symptom_lower], 1.0)]
    
    # Fuzzy match
    scores = []
    for vocab_symptom in VOCABULARY:
        score = similarity_score(symptom, vocab_symptom)
        scores.append((vocab_symptom, score))
    
    scores.sort(key=lambda x: -x[1])
    return scores[:top_n]


def analyze_all_symptoms():
    """Analyze all symptoms in template and generate report."""
    
    # Collect all symptoms
    all_mayo_symptoms = set()
    disease_symptoms = {}
    
    for disease, info in template.items():
        mayo = info.get("mayo_clinic_symptoms", [])
        disease_symptoms[disease] = mayo
        all_mayo_symptoms.update(mayo)
    
    print(f"Total unique mayo_clinic_symptoms: {len(all_mayo_symptoms)}")
    
    # Categorize by match quality
    exact_matches = []
    good_matches = []  # > 0.7 similarity
    poor_matches = []  # 0.4 - 0.7
    no_matches = []    # < 0.4
    
    symptom_analysis = {}
    
    for symptom in sorted(all_mayo_symptoms):
        matches = find_best_match(symptom)
        best_match, best_score = matches[0]
        
        symptom_analysis[symptom] = {
            "best_match": best_match,
            "score": best_score,
            "top_3": matches
        }
        
        if best_score == 1.0:
            exact_matches.append(symptom)
        elif best_score >= 0.7:
            good_matches.append((symptom, best_match, best_score))
        elif best_score >= 0.4:
            poor_matches.append((symptom, best_match, best_score))
        else:
            no_matches.append((symptom, best_match, best_score))
    
    # Generate report
    report = []
    report.append("=" * 80)
    report.append("SYMPTOM MAPPING ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    report.append(f"Total diseases in template: {len(template)}")
    report.append(f"Diseases with symptoms: {len([d for d,v in template.items() if v.get('mayo_clinic_symptoms')])}")
    report.append(f"Diseases without symptoms: {len([d for d,v in template.items() if not v.get('mayo_clinic_symptoms')])}")
    report.append(f"")
    report.append(f"Vocabulary size: {len(VOCABULARY)} symptoms")
    report.append(f"Total unique mayo_clinic_symptoms: {len(all_mayo_symptoms)}")
    report.append("")
    
    # Summary
    report.append("-" * 80)
    report.append("SUMMARY")
    report.append("-" * 80)
    report.append(f"  EXACT MATCHES (100%):     {len(exact_matches):4d} ({100*len(exact_matches)/len(all_mayo_symptoms):.1f}%)")
    report.append(f"  GOOD MATCHES (70-99%):    {len(good_matches):4d} ({100*len(good_matches)/len(all_mayo_symptoms):.1f}%)")
    report.append(f"  POOR MATCHES (40-69%):    {len(poor_matches):4d} ({100*len(poor_matches)/len(all_mayo_symptoms):.1f}%)")
    report.append(f"  NO MATCH (<40%):          {len(no_matches):4d} ({100*len(no_matches)/len(all_mayo_symptoms):.1f}%)")
    report.append("")
    report.append(f"  USABLE (exact + good):    {len(exact_matches) + len(good_matches):4d} ({100*(len(exact_matches) + len(good_matches))/len(all_mayo_symptoms):.1f}%)")
    report.append(f"  NEED ATTENTION:           {len(poor_matches) + len(no_matches):4d} ({100*(len(poor_matches) + len(no_matches))/len(all_mayo_symptoms):.1f}%)")
    report.append("")
    
    # Exact matches
    report.append("-" * 80)
    report.append(f"EXACT MATCHES ({len(exact_matches)})")
    report.append("-" * 80)
    for s in exact_matches[:20]:
        report.append(f"  [OK] {s}")
    if len(exact_matches) > 20:
        report.append(f"  ... and {len(exact_matches) - 20} more")
    report.append("")
    
    # Good matches
    report.append("-" * 80)
    report.append(f"GOOD MATCHES - 70-99% similarity ({len(good_matches)})")
    report.append("-" * 80)
    for s, m, score in sorted(good_matches, key=lambda x: -x[2]):
        report.append(f"  [{int(score*100):3d}%] \"{s}\" -> \"{m}\"")
    report.append("")
    
    # Poor matches
    report.append("-" * 80)
    report.append(f"POOR MATCHES - 40-69% similarity ({len(poor_matches)})")
    report.append("-" * 80)
    for s, m, score in sorted(poor_matches, key=lambda x: -x[2]):
        report.append(f"  [{int(score*100):3d}%] \"{s}\" -> \"{m}\"")
    report.append("")
    
    # No matches
    report.append("-" * 80)
    report.append(f"NO MATCH - <40% similarity ({len(no_matches)})")
    report.append("These symptoms have no good equivalent in your vocabulary")
    report.append("-" * 80)
    for s, m, score in sorted(no_matches, key=lambda x: -x[2]):
        report.append(f"  [{int(score*100):3d}%] \"{s}\" (closest: \"{m}\")")
    report.append("")
    
    # Diseases without symptoms
    report.append("-" * 80)
    report.append("DISEASES WITHOUT SYMPTOMS")
    report.append("-" * 80)
    for disease, info in template.items():
        if not info.get("mayo_clinic_symptoms"):
            report.append(f"  - {disease} ({info.get('current_samples', '?')} samples)")
    report.append("")
    
    # Per-disease analysis
    report.append("-" * 80)
    report.append("PER-DISEASE ANALYSIS (usable symptoms after mapping)")
    report.append("-" * 80)
    
    disease_usable = []
    for disease, info in template.items():
        mayo = info.get("mayo_clinic_symptoms", [])
        if not mayo:
            continue
        
        usable = 0
        for s in mayo:
            matches = find_best_match(s)
            if matches[0][1] >= 0.7:
                usable += 1
        
        pct = 100 * usable / len(mayo) if mayo else 0
        disease_usable.append((disease, len(mayo), usable, pct))
    
    # Sort by usable percentage
    for disease, total, usable, pct in sorted(disease_usable, key=lambda x: x[3]):
        status = "[!!]" if pct < 50 else "[OK]" if pct >= 80 else "[~~]"
        report.append(f"  {status} {disease}: {usable}/{total} usable ({pct:.0f}%)")
    
    report.append("")
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    return "\n".join(report), {
        "exact": exact_matches,
        "good": good_matches,
        "poor": poor_matches,
        "no_match": no_matches,
        "symptom_analysis": symptom_analysis
    }


if __name__ == "__main__":
    report_text, analysis = analyze_all_symptoms()
    
    # Print to console
    print(report_text)
    
    # Save report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"\n\nReport saved to: {report_path}")
    print("\nNOTE: This is READ-ONLY analysis. No data files were modified.")
