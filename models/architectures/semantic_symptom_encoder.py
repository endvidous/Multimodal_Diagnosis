"""
Semantic Symptom Encoder (Soft Evidence Version)

Encodes free-form symptom text into a 377-dimensional continuous
"symptom evidence" vector using sentence-transformer embeddings.

This encoder DOES NOT perform symptom detection.
It estimates latent evidence strength for each canonical symptom,
which is consumed by downstream ML models (e.g., LightGBM).

Key properties:
- Sentence-level encoding with top-k mean pooling
- Continuous evidence output (0–1), not binary
- MiniLM-friendly (low compute, stable)
- Lexical safety net for literal symptom mentions
"""

import re
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer


class SemanticSymptomEncoder:
    def __init__(
        self,
        model_name: str = "multi-qa-mpnet-base-dot-v1",
        symptom_vocab_path: Optional[str] = None,
        embeddings_cache_path: Optional[str] = None,
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.device = device

        # Resolve project root
        project_root = Path(__file__).parent.parent.parent

        if symptom_vocab_path is None:
            symptom_vocab_path = project_root / "data" / "symptom_vocabulary.json"
        if embeddings_cache_path is None:
            embeddings_cache_path = (
                project_root
                / "data"
                / "embeddings"
                / f"semantic_symptom_embeddings_{model_name}.npy"
            )

        self.symptom_vocab_path = Path(symptom_vocab_path)
        self.embeddings_cache_path = Path(embeddings_cache_path)

        # Load model
        print(f"[Encoder] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)

        # Load symptoms
        self.symptoms = self._load_symptoms()
        self.symptom_to_idx = {s: i for i, s in enumerate(self.symptoms)}
        self.idx_to_symptom = {i: s for i, s in enumerate(self.symptoms)}

        # Load or compute embeddings
        self.symptom_embeddings = self._load_or_compute_embeddings()

        print(f"[Encoder] Initialized with {len(self.symptoms)} symptoms")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _load_symptoms(self) -> List[str]:
        with open(self.symptom_vocab_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _normalize(self, text: str) -> str:
        return re.sub(r"[^a-z0-9 ]+", "", text.lower())

    def _split_into_sentences(self, text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text.strip())
        if not text:
            return []

        parts = re.split(
            r"[.!?;]\s*|\n+|,\s*(?:and|or)?\s*",
            text
        )
        parts = [p.strip() for p in parts if len(p.strip()) > 3]
        return parts if parts else [text]

    def _enrich_symptom(self, symptom: str) -> str:
        return ". ".join([
            symptom,
            f"I have {symptom}",
            f"I feel {symptom}",
            f"suffering from {symptom}",
            f"symptoms of {symptom}",
            f"my {symptom}"
        ])

    def _load_or_compute_embeddings(self) -> np.ndarray:
        if self.embeddings_cache_path.exists():
            emb = np.load(self.embeddings_cache_path)
            if emb.shape[0] == len(self.symptoms):
                print("[Encoder] Loaded cached symptom embeddings")
                return emb
            else:
                print("[Encoder] Cached embeddings mismatch, recomputing")

        print("[Encoder] Computing symptom embeddings (one-time)")
        enriched = [self._enrich_symptom(s) for s in self.symptoms]

        emb = self.model.encode(
            enriched,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        self.embeddings_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.embeddings_cache_path, emb)

        return emb

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def encode_symptoms(self, text: str, return_all_scores: bool = False) -> Dict[str, Any]:
        """
        Convert free-form symptom text into a 377-dim continuous
        symptom evidence vector (0–1).
        
        Args:
            text: Input symptom description
            return_all_scores: If True, includes a dict mapping all symptoms to their scores
        """
        if not text or not text.strip():
            return {
                "symptom_vector": np.zeros(len(self.symptoms), dtype=np.float32)
            }

        sentences = self._split_into_sentences(text)

        sent_emb = self.model.encode(
            sentences,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Sentence × Symptom similarity
        sims = sent_emb @ self.symptom_embeddings.T  # (S, 377)

        # Top-k mean pooling (k=2)
        k = min(2, sims.shape[0])
        topk = np.partition(sims, -k, axis=0)[-k:]
        pooled = topk.mean(axis=0)

        # MiniLM calibration
        evidence = np.clip(pooled - 0.25, 0.0, 1.0)
        evidence = evidence ** 1.5

        # Lexical safety net
        text_n = self._normalize(text)
        for i, symptom in enumerate(self.symptoms):
            if self._normalize(symptom) in text_n:
                evidence[i] = max(evidence[i], 0.9)
        
        response = {
            "symptom_vector": evidence.astype(np.float32)
        }
        
        if return_all_scores:
            response["all_scores"] = evidence

        return response

    def batch_encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode multiple texts. 
        """
        if not texts:
            return np.empty((0, len(self.symptoms)), dtype=np.float32)

        vectors = [
            self.encode_symptoms(t)["symptom_vector"]
            for t in texts
        ]
        return np.vstack(vectors)

    def get_similar_symptoms(self, symptom: str, top_k: int = 5):
        """
        Debug utility: find semantically similar symptoms.
        """
        if symptom in self.symptom_to_idx:
            q = self.symptom_embeddings[self.symptom_to_idx[symptom]]
        else:
            q = self.model.encode(
                symptom,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

        sims = self.symptom_embeddings @ q
        idxs = np.argsort(sims)[::-1][:top_k + 1]

        return [
            (self.idx_to_symptom[i], float(sims[i]))
            for i in idxs
            if self.idx_to_symptom[i] != symptom
        ][:top_k]

    def regenerate_embeddings(self):
        if self.embeddings_cache_path.exists():
            self.embeddings_cache_path.unlink()
        self.symptom_embeddings = self._load_or_compute_embeddings()

    def get_top_symptoms(self, symptom_vector: np.ndarray, top_k: int = 10, threshold: float = 0.0) -> List[tuple]:
        """
        Extract top-k symptoms with scores from an evidence vector.
        
        Args:
            symptom_vector: The evidence vector from encode_symptoms()
            top_k: Number of top symptoms to return
            threshold: Minimum score threshold (symptoms below this are excluded)
        
        Returns:
            List of (symptom_name, score) tuples, sorted by score descending
        """
        indices = np.argsort(symptom_vector)[::-1][:top_k]
        results = []
        for idx in indices:
            score = float(symptom_vector[idx])
            if score >= threshold:
                results.append((self.idx_to_symptom[idx], score))
        return results

    def get_symptom_score(self, symptom_vector: np.ndarray, symptom_name: str) -> float:
        """
        Get the score for a specific symptom from an evidence vector.
        
        Args:
            symptom_vector: The evidence vector from encode_symptoms()
            symptom_name: Name of the symptom to look up
            
        Returns:
            Score for the symptom (0.0 if not found)
        """
        if symptom_name in self.symptom_to_idx:
            return float(symptom_vector[self.symptom_to_idx[symptom_name]])
        return 0.0


# ----------------------------------------------------------------------
# Quick sanity test
# ----------------------------------------------------------------------

def quick_test():
    encoder = SemanticSymptomEncoder()

    tests = [
        "my head is killing me",
        "I can't catch my breath and my chest feels tight",
        "throwing up all morning and feeling nauseous",
        "burning sensation when I urinate",
        "I feel so tired and my joints ache all over",
        "stomach pain after eating, feels like heartburn",
        "runny nose and sore throat, probably a cold",
        "I've been feeling really anxious and can't sleep",
    ]

    for t in tests:
        vec = encoder.encode_symptoms(t)["symptom_vector"]
        top = np.argsort(vec)[-8:][::-1]
        print(f"\n📝 {t}")
        for i in top:
            print(f"  • {encoder.idx_to_symptom[i]}: {vec[i]:.3f}")


if __name__ == "__main__":
    quick_test()
