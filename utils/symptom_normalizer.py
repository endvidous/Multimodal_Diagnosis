"""
Symptom Normalizer: Utilities for normalizing and deduplicating symptoms.

This module provides functions to:
- Normalize symptoms to canonical form (singular, spell-corrected)
- Map synonyms to standard terms
- Validate and deduplicate symptom vocabularies
"""

import re
from typing import Optional

# Import constants from consts file
from utils.consts import TYPO_MAP, PLURAL_MAP, SYNONYM_MAP


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
    symptom = symptom.lower().strip().replace(',', '') 
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
