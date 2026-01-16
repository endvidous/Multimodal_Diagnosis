"""
Symptom Normalizer: Utilities for normalizing and deduplicating symptoms.

This module provides functions to:
- Normalize symptoms to canonical form (singular, spell-corrected)
- Map synonyms to standard terms
- Validate and deduplicate symptom vocabularies
"""

import re
from typing import Optional

# Common typos found in medical symptom data
# Common typos found in medical symptom data
TYPO_MAP = {
    # Spelling errors
    'vomitting': 'vomiting',
    'apetite': 'appetite',
    'neusea': 'nausea',
    'dizzy': 'dizziness',
    'weakeness': 'weakness',
    'stiffeness': 'stiffness',
    'numbess': 'numbness',
    'paleness': 'pallor',
    'tireness': 'tiredness',
    'slowhealing': 'slow healing',
    'thirsty': 'thirst',
    'fefver': 'fever',
    'burpin': 'burping',
    'itchness': 'itchiness',
    
    # Phrase typos
    'loss of consiousness': 'loss of consciousness',
    'loss of apetite': 'loss of appetite',
    'lack of apetite': 'loss of appetite',
    'nausea and vomitting': 'nausea and vomiting',
    'ringing in ears': 'ringing in ear',
    ''
    
    # Data merge artifacts
    'regurgitation.1': 'regurgitation',
}

# Plural to singular mappings for medical symptoms
# These are irregular or domain-specific plurals that simple rules won't catch
PLURAL_MAP = {
    'headaches': 'headache',
    'rashes': 'rash',
    'nosebleeds': 'nosebleed',
    'seizures': 'seizure',  # Keep as singular
    'chills': 'chills',  # Exception: keep as plural (it's a mass noun)
    'muscle aches': 'muscle ache',
    'body aches': 'body ache',
    'ringing in ears': 'ringing in ear',
    'swelling of feets': 'swelling of feet',
    'bloody stools': 'bloody stool',
    'irregular heartbeats': 'irregular heartbeat',
    'swelling of eyelids': 'swelling of eyelid',
    'numbness in arms': 'numbness in arm',
    'numbness in legs': 'numbness in leg',
    'swollen glands': 'swollen lymph nodes',
    'swollen lymph glands': 'swollen lymph nodes',
}

# Synonym mappings - map variants to canonical form
SYNONYM_MAP = {
    # Pain locations
    'belly pain': 'abdominal pain',
    'stomach pain': 'abdominal pain',
    'tummy pain': 'abdominal pain',
    'abdominal distention': 'abdominal pain', # Often overlaps
    'abdominal cramps': 'abdominal pain',
    'stomach cramps': 'abdominal pain',
    
    # Fatigue variants
    'tiredness': 'fatigue',
    'extreme tiredness': 'fatigue',
    'lethargy': 'fatigue',
    'feeling tired': 'fatigue',
    'feeling weak': 'weakness',
    'extreme fatigue': 'fatigue',
    
    # Voice/throat
    'hoarseness': 'hoarse voice',
    
    # Weight changes
    'losing weight': 'weight loss',
    'unexplained weight loss': 'weight loss',
    'unintentional weight loss': 'weight loss',
    'recent weight loss': 'weight loss',
    'unexplained weight gain': 'weight gain',
    'failure to gain weight': 'poor weight gain',
    
    # Consciousness
    'fainting': 'loss of consciousness',
    
    # Bloating
    'stomach bloating': 'bloating',
    'abdominal distention': 'bloating', # Can map to bloating or abdominal pain, bloating is more specific if present
    'feeling bloated': 'bloating',
    'belching': 'bloating', # Related
    'swollen abdomen': 'swollen abdomen', # Keep specific if present
    
    # Appetite
    'loss of apetite': 'loss of appetite',  # Fix typo AND standardize
    'lack of apetite': 'loss of appetite',
    'lack of appetite': 'loss of appetite',
    'poor apetite': 'loss of appetite',
    'poor appetite': 'loss of appetite',
    'decreased appetite': 'loss of appetite',
    
    # Nausea/vomiting
    'nausea and vomitting': 'nausea and vomiting',
    'vomiting blood': 'vomiting blood',
    'throwing up': 'vomiting',
    
    # Swallowing
    'difficulty in swallowing': 'difficulty swallowing',
    'trouble swallowing': 'difficulty swallowing',
    'pain when swallowing': 'difficulty swallowing', 
    
    # Speech
    'trouble speaking': 'difficulty speaking',
    'slurring words': 'slurred speech',
    'slow speech': 'slurred speech',
    
    # Vision
    'blurry vision': 'blurred vision',
    'blurred vision': 'blurred vision', # Ensure target exists
    'trouble with vision': 'vision problems',
    'vision less clear': 'vision loss',
    'light sensitivity': 'sensitivity to light',
    'redness of eye': 'eye redness',
    'eye inflammation': 'eye redness',
    'pink eye': 'eye redness',
    'involuntary eye movements': 'eye moves abnormally',
    
    # Skin
    'changes in skin color': 'skin color changes',
    'skin flushing': 'flushing',
    'itching': 'itching of skin',
    'itchy skin': 'itching of skin',
    'yellowing of skin': 'jaundice',
    'yellow skin': 'jaundice',
    'pale skin': 'pallor',
    
    # Sensation/Coordination
    'lack of coordination': 'loss of coordination',
    'poor coordination': 'loss of coordination',
    'loss of touch': 'loss of sensation',
    'reduced pain sensation': 'loss of sensation',
    'loss of balance': 'problems with balance',
    'dizzy': 'dizziness',
    
    # Heart
    'fast heart beat': 'fast heart rate',
    'fast heartbeats': 'fast heart rate',
    'pounding heart': 'heart palpitations',
    'heart failure': 'heart palpitations', # Symptom approx
    
    # Swelling/Lumps
    'swelling of legs': 'leg swelling',
    'swollen legs': 'leg swelling',
    'swelling in legs': 'leg swelling',
    'swelling of feet': 'foot or toe swelling',
    'swollen feet': 'foot or toe swelling',
    'swollen toes': 'foot or toe swelling',
    'swelling of hands': 'hand or finger swelling',
    'swollen fingers': 'hand or finger swelling',
    'facial swelling': 'swelling', 
    'swollen tonsils': 'swollen or red tonsils',
    'lump on vulva': 'mass on vulva',
    'swollen lymph glands': 'swollen lymph nodes',
    
    # Other
    'pain during sex': 'pain during intercourse',
    'severe headache': 'headache',
    'pain in joint': 'joint pain',
    'muscle tightness': 'muscle stiffness or tightness',
    'blood in feces': 'blood in stool',
    'bloody stool': 'blood in stool',
    'muscle or joint pain': 'muscle pain',
    'dry mouth and lips': 'dry mouth',
    'gas pain': 'gas',
    'loss of smell': 'disturbance of smell or taste',
    'discomfort in chest': 'chest pain',
    'heavy menstrual periods': 'heavy menstrual flow',
    'irregular periods': 'irregular menstrual cycles', # or irregular menstruation
    'high bmi': 'obesity', # if obesity is in vocab? Or weight gain.
}


def normalize_symptom(symptom: str, apply_synonyms: bool = True) -> str:
    """
    Normalize a symptom string to its canonical form.
    
    Steps:
    1. Lowercase and strip whitespace
    2. Remove data artifacts (e.g., ".1" suffix from pandas)
    3. Fix common typos
    4. Convert plural to singular (where appropriate)
    5. Optionally map synonyms to canonical forms
    
    Args:
        symptom: The symptom string to normalize
        apply_synonyms: Whether to apply synonym mapping
        
    Returns:
        Normalized symptom string
    """
    if not symptom:
        return ""
    
    # Step 1: Basic cleanup
    symptom = symptom.lower().strip()
    
    # Step 2: Remove pandas/data merge artifacts
    symptom = re.sub(r'\.\d+$', '', symptom)
    
    # Step 3: Fix typos (check full string first, then individual words)
    if symptom in TYPO_MAP:
        symptom = TYPO_MAP[symptom]
    else:
        for typo, correct in TYPO_MAP.items():
            if typo in symptom:
                symptom = symptom.replace(typo, correct)
    
    # Step 4: Plural to singular conversion
    if symptom in PLURAL_MAP:
        symptom = PLURAL_MAP[symptom]
    
    # Step 5: Synonym mapping
    if apply_synonyms and symptom in SYNONYM_MAP:
        symptom = SYNONYM_MAP[symptom]
    
    return symptom


def find_similar_symptoms(symptom: str, vocabulary: list[str], 
                          threshold: float = 0.8) -> list[tuple[str, float]]:
    """
    Find symptoms in vocabulary that are similar to the given symptom.
    
    Uses token overlap (Jaccard similarity) for comparison.
    
    Args:
        symptom: Symptom to search for
        vocabulary: List of symptoms to search in
        threshold: Minimum similarity score (0-1)
        
    Returns:
        List of (symptom, score) tuples sorted by score descending
    """
    from difflib import SequenceMatcher
    
    def tokenize(text: str) -> set:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return set(text.split())
    
    def similarity(a: str, b: str) -> float:
        tokens_a = tokenize(a)
        tokens_b = tokenize(b)
        
        if not tokens_a or not tokens_b:
            return 0.0
        
        overlap = len(tokens_a & tokens_b)
        total = len(tokens_a | tokens_b)
        jaccard = overlap / total if total > 0 else 0
        
        seq_ratio = SequenceMatcher(None, a.lower(), b.lower()).ratio()
        
        return 0.6 * jaccard + 0.4 * seq_ratio
    
    matches = []
    normalized = normalize_symptom(symptom, apply_synonyms=False)
    
    for vocab_symptom in vocabulary:
        score = similarity(normalized, vocab_symptom.lower())
        if score >= threshold:
            matches.append((vocab_symptom, score))
    
    return sorted(matches, key=lambda x: -x[1])


def validate_vocabulary(symptoms: list[str], 
                        apply_synonyms: bool = True,
                        verbose: bool = False) -> tuple[list[str], dict]:
    """
    Validate and deduplicate a symptom vocabulary.
    
    Args:
        symptoms: List of symptom strings
        apply_synonyms: Whether to merge synonyms
        verbose: Whether to print detailed changes
        
    Returns:
        Tuple of (cleaned_symptoms, changes_dict)
        changes_dict contains: removed, merged, typo_fixed
    """
    changes = {
        'removed': [],      # Exact duplicates removed
        'merged': [],       # Synonyms merged
        'typo_fixed': [],   # Typos corrected
        'original_count': len(symptoms),
    }
    
    seen = {}  # normalized -> original
    
    for symptom in symptoms:
        original = symptom
        normalized = normalize_symptom(symptom, apply_synonyms=apply_synonyms)
        
        if not normalized:
            continue
        
        # Track changes
        if original.lower() != normalized:
            if original.lower() in TYPO_MAP or any(t in original.lower() for t in TYPO_MAP):
                changes['typo_fixed'].append((original, normalized))
            elif apply_synonyms and original.lower() in SYNONYM_MAP:
                changes['merged'].append((original, normalized))
        
        if normalized in seen:
            changes['removed'].append((original, f"duplicate of '{seen[normalized]}'"))
            if verbose:
                print(f"  Removed duplicate: '{original}' (same as '{seen[normalized]}')")
        else:
            seen[normalized] = original
            if verbose and original.lower() != normalized:
                print(f"  Normalized: '{original}' -> '{normalized}'")
    
    changes['final_count'] = len(seen)
    
    # Return normalized forms sorted alphabetically
    return sorted(seen.keys()), changes


def get_canonical_form(symptom: str) -> str:
    """
    Get the canonical (standard) form of a symptom.
    
    This is a convenience function that applies full normalization.
    
    Args:
        symptom: Symptom string
        
    Returns:
        Canonical form of the symptom
    """
    return normalize_symptom(symptom, apply_synonyms=True)


# Add new mappings dynamically
def add_typo_mapping(typo: str, correct: str) -> None:
    """Add a new typo -> correct mapping."""
    TYPO_MAP[typo.lower()] = correct.lower()


def add_synonym_mapping(synonym: str, canonical: str) -> None:
    """Add a new synonym -> canonical mapping."""
    SYNONYM_MAP[synonym.lower()] = canonical.lower()


def add_plural_mapping(plural: str, singular: str) -> None:
    """Add a new plural -> singular mapping."""
    PLURAL_MAP[plural.lower()] = singular.lower()
