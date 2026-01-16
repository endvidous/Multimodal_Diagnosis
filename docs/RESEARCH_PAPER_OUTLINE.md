# Research Paper Outline

## Semantic and Demographic-Aware Disease Prediction from Symptoms

**Target**: College Conference Presentation

---

## Abstract (150 words)

We present a symptom-to-disease prediction system that combines:

1. **Semantic symptom encoding** using sentence transformers to understand free-text descriptions
2. **Hierarchical classification** (category → disease) for 667 diseases across 14 categories
3. **Demographic features** (age, sex) to improve diagnostic accuracy

Key results on 224K samples:

- **Hierarchical Ensemble:** Top-1: 84.88%, Top-3: 96.08	%
- **Baseline (Flat):** Top-1: 80.2%, Top-5: 91.5%
- Demographics add +2.8% improvement to base models
- Outperforms logistic regression and random forest baselines

---

## 1. Introduction

- **Problem**: Symptom checkers require structured input; users describe symptoms in natural language
- **Gap**: Most systems don't leverage patient demographics effectively
- **Contribution**: End-to-end system from free-text → diagnosis with demographic priors

---

## 2. Related Work

| System         | Approach          | Diseases | Top-1 Acc       |
| -------------- | ----------------- | -------- | --------------- |
| Ada Health     | Rule-based + ML   | ~1000    | 51%             |
| Babylon        | Bayesian networks | ~500     | 60%             |
| Isabel         | Knowledge graph   | ~6000    | 48%             |
| **Ours** | Semantic + ML     | 667      | **77.3%** |

---

## 3. Methods

### 3.1 Semantic Symptom Encoder

- MiniLM-L6-v2 sentence embeddings (384-dim)
- 377 canonical symptoms with enriched descriptions
- **Sentence-level encoding** to prevent symptom dilution
- Similarity threshold-based matching

### 3.2 Hierarchical Classification

- Stage 1: Category classifier (14 classes) - 90.5% accuracy
- Stage 2: Specialist Disease classifiers (14 models)
- Stage 3: **Probabilistic Ensemble Routing** (Top-3 Categories)
- Achieves **84.88% Top-1 Accuracy** (vs 80.2% Flat)

### 3.3 Demographic Features

- Age (normalized 0-1) + Sex (binary)
- Trained separate model with 482 features
- Statistical significance via McNemar's test

### 3.4 Data Processing Pipeline

We established a strict sequential pipeline to ensure data integrity and model robustness:

1.  **Vocabulary Standardization**:
    - *Rationale*: Raw medical text is noisy. "Vomitting" and "Vomiting" must be treated as the same feature to prevent signal dilution. We merged synonyms and fixed typos *before* any downstream processing.
    
2.  **Dataset Consolidation**:
    - *Rationale*: Merging duplicate columns (e.g., separate features for `headache` and `headaches`) recovers lost information. If a patient had `headaches=1` but `headache=0`, a naive model might miss the signal. Our MAX-merge strategy ensures ~42% of rows gained feature density.

3.  **Synthetic Augmentation**:
    - *Rationale*: Real-world data is heavily imbalanced. Rare diseases (e.g., *Progeria*) had <10 samples. We synthesized samples using authorized symptom lists (Mayo Clinic) to ensure the model learns robust decision boundaries for all 667 classes, not just common ones.

4.  **Two-Stage Training**:
    - *Rationale*: We decoupled feature learning (Semantic Encoder) from classification. This allows the encoder to focus purely on "understanding symptoms" without overfitting to specific disease prevalences.

---

## 4. Dataset

- **Source**: Augmented symptom-disease dataset
- **Size**: 224K samples, 667 diseases, 14 categories
- **Features**: 455 canonical symptoms + 2 demographic
- **Split**: 80% train, 10% val, 10% test

### 4.1 Data Preprocessing & Cleaning

#### Symptom Vocabulary Normalization

We implemented a comprehensive symptom normalization pipeline to ensure data consistency:

1. **Typo Correction**: Fixed common medical typos
   - `vomitting` → `vomiting`
   - `apetite` → `appetite`
   - `neusea` → `nausea`

2. **Singular/Plural Standardization**: Converted to singular forms
   - `headaches` → `headache`
   - `rashes` → `rash`
   - `nosebleeds` → `nosebleed`

3. **Synonym Consolidation**: Mapped semantic equivalents to canonical forms
   - `belly pain`, `stomach pain` → `abdominal pain`
   - `tiredness`, `lethargy`, `extreme tiredness` → `fatigue`
   - `hoarseness` → `hoarse voice`
   - `losing weight`, `unexplained weight loss` → `weight loss`

4. **Data Artifact Removal**: Cleaned pandas merge artifacts
   - `regurgitation.1` → `regurgitation`

#### Duplicate Column Merging

The augmented dataset contained 17 groups of duplicate symptom columns (e.g., `vomiting` and `vomitting` as separate features). We merged these using a MAX strategy:

- If **any** duplicate column has value 1 → merged result = 1
- This preserves all positive symptom signals (no information loss)
- Reduced feature space from 481 to 456 unique symptoms

**Impact**: 93,520 rows (41.76%) gained additional symptom signals through consolidation.

#### Rare Disease Augmentation

Diseases with <25 training samples were augmented using Mayo Clinic symptom data:
- Manual curation of symptoms from authoritative medical sources
- Symptom mapping to standardized vocabulary using fuzzy matching
- Synthetic sample generation with random symptom subsets

---

## 5. Experiments

### 5.1 Semantic Encoder Evaluation

- Accuracy on colloquial symptom phrases
- Sentence-level vs whole-text comparison

### 5.2 Classification Performance

- Top-k accuracy (k=1,3,5,10)
- Per-category confusion matrix
- Macro F1 scores

### 5.3 Demographics Impact

- Per-disease improvement analysis
- Statistical significance testing

### 5.4 Baseline Comparisons

- Logistic Regression
- Random Forest (100 trees)

---

## 6. Results

### Table 1: Main Results

| Model                           | Top-1            | Top-3            | Top-5            |
| ------------------------------- | ---------------- | ---------------- | ---------------- |
| Symptoms Only                   | 77.4%            | 88.5%            | 90.9%            |
| + Demographics                  | 80.2%            | 89.4%            | 91.5%            |
| **Hierarchical Ensemble** | **84.88%** | **96.08%** | **97.99%** |

### Table 2: Baseline Comparison

| Model               | Accuracy         | Training Time |
| ------------------- | ---------------- | ------------- |
| Logistic Regression | ~79.01%          | 193.3s        |
| Random Forest       | ~66.69%          | 21.4s         |
| LightGBM (Ours)     | **80.20%** | 60s           |

---

## 7. Discussion

### 7.1 Interpreting Performance Metrics

- **Context of +1.5% Improvement**: A +1.5% gain over Logistic Regression is significant in medical diagnosis. Linear models perform well on structured data, but fail on the complex edge cases that our model captures.
- **The "Model" is the Pipeline**: The core innovation is the **Semantic Encoder**, which translates raw text (e.g., "my head hurts") into interpretable features. Standard baselines like Logistic Regression require structured input and cannot function on natural language directly.
- **Hierarchical Superiority**: The Hierarchical Ensemble (84.88%) significantly outperforms the Flat Demographic model (80.20%). By training specialist models, we reduce the decision space for each classifier, allowing them to learn subtler distinctions between similar diseases (e.g., *Flu* vs *Common Cold* within *Infectious Diseases*).
- **Soft Routing Robustness**: Our "Top-3 Probabilistic Routing" strategy (96.08% Top-3) mitigates the risk of cascading errors where a wrong category prediction would otherwise lead to failure.
- **Comparison to Random Forest**: We achieved a **+18.2% improvement** over Random Forest with the ensemble.
- **Top-5 Accuracy (98.0%)**: The correct disease is in the top 5 candidates 98% of the time, making this a highly reliable filter for doctors.

### 7.2 Strengths & Limitations

**Strengths:**

- Semantic understanding bridges vocabulary gap
- Demographics help for age/sex-specific diseases
- Interpretable two-stage prediction

**Limitations:**

- Negation not explicitly handled ("I don't have fever")
- Limited to symptom modality (no images/labs)
- Synthetic augmentation for rare diseases

---

## 8. Conclusion

We demonstrated a practical symptom-to-disease system achieving 77.3% Top-1 accuracy on 667 diseases. Key innovations:

1. Sentence-level semantic encoding
2. Demographic-aware prediction (+2.1%)
3. Hierarchical classification for interpretability

**Future Work:**

- Multimodal extension (X-rays, blood reports)
- Clinical validation study
- Deployment as doctor-facing tool

---

## Figures to Include

1. **System Architecture** - Flow diagram: Text → Encoder → Classifier → Diagnosis
2. **Model Comparison** - Bar chart from evaluation.ipynb
3. **Confusion Matrix** - Category-level heatmap
4. **Demographics Impact** - Per-disease improvement chart

---

## Presentation Tips (College Conference)

- Keep slides visual, minimal text
- Demo the semantic encoder live ("my head is killing me" → headache)
- Highlight the +2.1% demographics improvement as key finding
- Compare to well-known apps (WebMD, Ada) for context
- Prepare for questions about clinical validation
