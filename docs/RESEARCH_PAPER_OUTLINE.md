# Research Paper Outline

## Semantic and Demographic-Aware Disease Prediction from Symptoms

**Target**: College Conference Presentation

---

## Abstract (200 words)

We present a hierarchical symptom-to-disease prediction system combining semantic encoding with demographic-aware classification. Our approach addresses key challenges in automated medical diagnosis: bridging the gap between colloquial and clinical terminology, handling severe class imbalance in rare diseases, and incorporating patient demographics.

The system employs:
1. **Semantic symptom encoding** using multi-qa-mpnet-base-dot-v1 transformers (768-dim) to map free-text descriptions to 480 standardized symptoms
2. **Hierarchical classification** with specialist models for 667 diseases across 14 categories
3. **Demographic integration** (age, sex) for improved diagnostic accuracy

**Evaluation on 207K samples** (including synthetic augmentation for 135 rare diseases):
- Hierarchical Ensemble: **86.40%** Top-1, **97.91%** Top-5 accuracy
- Flat Baseline: 81.06% Top-1, 91.26% Top-5 accuracy  
- Relative improvement: **+5.34%** over flat baseline, **+19.7%** over Random Forest
- Demographics contribute **+2.9%** improvement

> [!IMPORTANT]
> **Evaluation Note**: Results obtained on partially synthetic dataset. Component testing suggests real-world end-to-end performance may be lower (estimated 65-75%). Clinical validation on real patient data is required before deployment.

---

## 1. Introduction

- **Problem**: Symptom checkers require structured input; users describe symptoms in natural language
- **Gap**: Most systems don't leverage patient demographics effectively
- **Contribution**: End-to-end system from free-text → diagnosis with demographic priors

---

## 2. Related Work

Prior work in automated symptom-based diagnosis spans several approaches:

**Rule-based systems**: Early symptom checkers used hand-crafted rules mapping symptoms to diseases, but suffered from maintenance burden and limited coverage.

**Probabilistic models**: Bayesian networks and similar approaches model symptom-disease relationships probabilistically, but require explicit probability elicitation.

**Machine learning approaches**: Recent work applies classical ML (Random Forest, SVM) and deep learning to symptom classification, typically requiring structured symptom input.

**Key gap**: Most existing systems require structured symptom input rather than free-text. Our contribution focuses on bridging colloquial symptom descriptions to clinical terminology via semantic encoding.

---

## 3. Methods

### 3.1 Semantic Symptom Encoder

- multi-qa-mpnet-base-dot-v1 sentence embeddings (768-dim)
- 458 canonical symptoms with enriched descriptions
- **Sentence-level encoding** to prevent symptom dilution
- Similarity threshold-based matching

### 3.2 Hierarchical Classification

- Stage 1: Category classifier (14 classes) - 92.9% accuracy
- Stage 2: Specialist Disease classifiers (14 models)
- Stage 3: **Probabilistic Ensemble Routing** (Top-3 Categories)
- Achieves **86.40% Top-1 Accuracy** (vs 81.06% Flat)

### 3.3 Demographic Features

- Age (normalized 0-1) + Sex (binary)
- Trained separate model with 482 features (480 symptoms + 2 demographic)
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
- **Size**: 207K samples, 667 diseases, 14 categories
- **Features**: 480 symptom features + 2 demographic
- **Split**: 80% train, 10% val, 10% test

### 4.1 Main Dataset

The dataset was constructed from a master list of **773 potential diseases** derived from medical ontologies and raw symptom data.

- **Initial Collection**: 773 diseases in raw dataset
- **Data Pruning Criteria**:
  - **Exclusion of Non-Predictable Conditions**: Removed diseases primarily diagnosed via trauma, imaging, or lab tests rather than symptom patterns (e.g., "open wound due to trauma")
  - **Data Sufficiency**: Filtered diseases with insufficient symptom information (<4 symptoms)
- **Processed Statistics**:
  - **Processed Dataset**: 630 unique diseases retained after cleaning
  - **Final Modeling Target**: **615 diseases** selected for training (meeting minimum sample thresholds after augmentation)
- **Final Dataset**:
  - **Total Samples**: 207,387 (after filtering)
  - **Features**: 480 symptom features + 2 demographic features
  - **Symptom Vocabulary**: 458 canonical symptoms (mapped from colloquial terms)t

### 4.2 Data Preprocessing & Cleaning

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
- Reduced feature space from 481 to 480 unique symptoms after merging duplicates

**Impact**: 93,520 rows (41.76%) gained additional symptom signals through consolidation.

#### Rare Disease Augmentation Pipeline

To address severe class imbalance, we implemented a multi-stage augmentation pipeline:

**Stage 1: Disease Selection**
- Identified 135 diseases with fewer than 5 training samples
- Manual curation of symptom lists from authoritative medical websites (Mayo Clinic, Cleveland Clinic, WebMD, etc.)
- 11 diseases could not be augmented due to insufficient reliable symptom information

**Stage 2: Symptom Collection & Normalization**
- Manually entered symptoms from web sources into structured format
- Applied vocabulary normalization to fix typos and inconsistencies in manually entered data
- Mapped symptoms to standardized vocabulary using fuzzy matching (>85% similarity threshold)

**Stage 3: Synthetic Sample Generation**
- Generated synthetic samples by randomly selecting symptom subsets (50-80% of disease symptoms)
- Ensured minimum 25 samples per disease for adequate model training
- Preserved symptom co-occurrence patterns from source descriptions

**Stage 4: Demographic Augmentation**
- Applied demographic features (age, sex) based on disease epidemiology
- Created separate dataset version with demographics for comparative evaluation

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
| Symptoms Only                   | 78.5%            | 88.5%            | 89.6%            |
| + Demographics                  | 81.1%            | 89.4%            | 91.3%            |
| **Hierarchical Ensemble** | **86.40%** | **96.08%** | **97.91%** |

### Table 2: Baseline Comparison

| Model               | Accuracy         | Training Time |
| ------------------- | ---------------- | ------------- |
| Logistic Regression | ~79.01%          | 193.3s        |
| Random Forest       | ~66.69%          | 21.4s         |
| LightGBM (Ours)     | **81.06%** | 60s           |

---

## 7. Discussion

### 7.1 Interpreting Performance Metrics

- **Context of +2.0% Improvement**: A +2.0% gain over Logistic Regression is significant in medical diagnosis. Linear models perform well on structured data, but fail on the complex edge cases that our model captures.
- **The "Model" is the Pipeline**: The core innovation is the **Semantic Encoder**, which translates raw text (e.g., "my head hurts") into interpretable features. Standard baselines like Logistic Regression require structured input and cannot function on natural language directly.
- **Hierarchical Superiority**: The Hierarchical Ensemble (86.40%) significantly outperforms the Flat Demographic model (81.06%). By training specialist models, we reduce the decision space for each classifier, allowing them to learn subtler distinctions between similar diseases (e.g., *Flu* vs *Common Cold* within *Infectious Diseases*).
- **Soft Routing Robustness**: Our "Top-3 Probabilistic Routing" strategy (96.08% Top-3) mitigates the risk of cascading errors where a wrong category prediction would otherwise lead to failure.
- **Comparison to Random Forest**: We achieved a **+19.7% improvement** over Random Forest with the ensemble.
- **Top-5 Accuracy (97.9%)**: The correct disease is in the top 5 candidates 97.9% of the time, making this a highly reliable filter for doctors.

### 7.2 Strengths

- **Semantic Understanding**: Bridges vocabulary gap between colloquial and clinical terminology
- **Demographic Integration**: Improves prediction for age/sex-specific diseases (+2.9% improvement)
- **Hierarchical Architecture**: Interpretable two-stage prediction with specialist models
- **Class Imbalance Solution**: Demonstrates effective synthetic augmentation for rare diseases
- **Relative Improvement**: +5.3% over flat baseline, +19.7% over Random Forest

### 7.3 Limitations and Evaluation Considerations

> [!WARNING]
> **Critical Evaluation Limitations**: This study uses a partially synthetic dataset for evaluation. Real-world clinical validation is required before deployment.

**Dataset Composition and Synthetic Data:**
- Approximately 135 rare diseases (out of 667 total) were augmented with synthetic samples generated from web-sourced symptom descriptions
- Synthetic samples may not fully capture the complexity and variability of real patient presentations
- 11 diseases excluded due to insufficient reliable symptom information
- Test set contains both real and synthetic samples, potentially inflating accuracy estimates

**Evaluation Methodology Gaps:**
- **Component vs. End-to-End Performance**: The reported 86.4% Top-1 accuracy assumes correctly extracted symptom features (480-dimensional vectors)
- **Semantic Encoder Accuracy**: Preliminary testing shows ~80% accuracy on colloquial symptom phrase matching, suggesting real-world end-to-end performance would be lower (estimated 65-75%)
- **No Cross-Dataset Validation**: Model evaluated only on held-out portion of the same augmented dataset
- **Missing Clinical Validation**: No evaluation by medical professionals or on real electronic health records

**Comparison Validity:**
- Direct comparison with commercial systems (Ada Health: 51%, Babylon: 60%) should be interpreted cautiously:
  - Commercial systems evaluated on real clinical data vs. our partially synthetic test set  
  - Different disease coverage (1000+ vs. 615 diseases)
  - Evaluation methodologies and symptom input formats may differ significantly
  - Our dataset size (207K samples) is smaller and more controlled

**Technical Limitations:**
- **Negation Handling**: System does not explicitly handle negated symptoms (e.g., "I don't have fever")
- **Modality**: Limited to symptom-based diagnosis; does not incorporate imaging, lab results, or vital signs
- **Symptom Extraction**: Relies on sentence-transformer similarity; may miss domain-specific medical nuances
- **Temporal Information**: Does not model symptom onset timing or disease progression

**Generalization Concerns:**
- Training data primarily from English-language medical websites
- May not generalize to different populations, healthcare settings, or languages
- Disease prevalence in dataset may not reflect real-world epidemiology

---

## 8. Conclusion

We present a novel hierarchical approach to symptom-based disease prediction that demonstrates the potential of combining semantic understanding with demographic information. Our system achieves **86.4% Top-1 accuracy** on a partially synthetic dataset of 667 diseases, representing a **+5.34% improvement** over flat classification baselines and **+19.7% over Random Forest**.

**Key Contributions:**
1. **Semantic Bridge**: Multi-qa-mpnet-based encoder maps colloquial symptom descriptions to clinical features
2. **Hierarchical Specialists**: Two-stage classification with category-specific models improves accuracy and interpretability
3. **Demographic Integration**: Age/sex features provide meaningful improvement (+2.9%) for relevant diseases
4. **Class Imbalance Solution**: Demonstrate effective synthetic augmentation methodology for rare diseases

**Critical Limitations:**
While our results are promising on controlled data, several gaps remain before clinical deployment:
- Evaluation uses partially synthetic test data (~20% of diseases augmented)
- Component testing suggests real-world end-to-end accuracy may be 65-75%
- No validation on independent clinical datasets or by medical professionals
- Limited to English-language symptom descriptions

**Research Value:**
This work establishes a methodological framework for semantic symptom understanding and hierarchical disease classification. The relative improvements over baselines (+5.34%) demonstrate the value of our architectural choices, even if absolute accuracy requires clinical validation to establish real-world utility.

## 9. Future Work

**Immediate Priorities:**

1. **Clinical Validation Study**
   - Evaluate on real electronic health record data
   - Partner with medical institutions for prospective testing
   - Measure true end-to-end performance (free-text input → diagnosis)
   - Compare against physician diagnostic accuracy

2. **Cross-Dataset Evaluation**
   - Test on public medical datasets (e.g., MIMIC-III symptom notes)
   - Evaluate generalization to different populations
   - Measure performance degradation on out-of-distribution data

3. **Semantic Encoder Improvements**
   - Fine-tune multi-qa-mpnet on medical symptom corpora
   - Add domain-specific training data from clinical notes
   - Improve negation handling and temporal reasoning
   - Achieve >90% symptom extraction accuracy

**Medium-Term Extensions:**

4. **Multimodal Integration**
   - Incorporate vital signs (temperature, blood pressure, heart rate)
   - Add blood test result interpretation
   - Integrate medical imaging (X-rays, CT scans) for relevant diseases

5. **Enhanced Interpretability**
   - Implement attention mechanisms to highlight key symptoms
   - Generate natural language explanations for predictions
   - Provide differential diagnosis with confidence-calibrated probabilities

6. **Robustness Improvements**
   - Handle multi-lingual symptom descriptions
   - Model disease progression and symptom timing
   - Address adversarial inputs and edge cases

**Long-Term Vision:**

7. **Clinical Decision Support System**
   - Deploy as physician-facing tool (not patient-facing)
   - Integrate with hospital EHR systems
   - Continuous learning from expert feedback
   - Regular model retraining with new clinical data

8. **Regulatory Approval**
   - FDA clearance as clinical decision support software
   - Compliance with medical device regulations
   - Rigorous safety and efficacy validation

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
