# End-to-End Model Training Workflow

This document outlines the complete valid execution order for the Multimodal Diagnosis project, from raw data to a trained model.

## 1. Data Cleaning & Preparation
**Goal**: Normalize symptoms, fix typos, and merge duplicate columns.

- **Step 1: Clean Data**  
  Run the consolidated cleaning notebook. This handles both vocabulary normalization and dataset standardization.
  - Open `notebooks/2.data_cleaning.ipynb`
  - **Run All Cells**.
  - *Output*: Updates `data/symptom_vocabulary.json` and all CSV files in `data/processed/`.

## 2. Augmentation (Crucial)

**Goal**: Fix class imbalance. The raw dataset has 100+ diseases with only ~1 sample. Models cannot learn from 1 sample.

- **Step 3: Map Rare Diseases** (One-time setup)
  Maps symptoms from `data/rare_diseases_symptoms_template.json` to your vocabulary.
  *(This is now automated with the `--auto` flag)*

  ```bash
  python scripts/symptom_mapper.py --auto --remap
  ```
- **Step 4: Generate Synthetic Data**  
  Creates synthetic patient records for rare diseases.
  - Open `notebooks/4.data_augment.ipynb`
  - **Run All Cells**.
  
  **Result**: Creates `data/processed/symptoms/symptoms_to_disease_augmented.csv`.

## 3. Model Training

**Goal**: Train the Semantic Encoder and Disease Classifiers using the CLEAN, AUGMENTED data.

- **Step 5: Train Semantic Encoder**Learns to associate natural language with clinical features.

  - Open `models/training/train_semantic_encoder_unified.ipynb`
  - **Run All Cells**.
  - *Output*: Saves `semantic_encoder.joblib` and mappings.
- **Step 6: Train Classifiers**Trains the Random Forest/XGBoost models.

  - Open `models/training/train_classifiers_unified.ipynb`
  - **Run All Cells**.
  - *Output*: Saves `disease_classifier.joblib`.

## 4. Evaluation

**Goal**: Verify model performance.

- **Step 7: Evaluate**Calculates Top-1 and Top-3 accuracy on the test set.
  - Open `notebooks/5.evaluation.ipynb`
  - **Run All Cells**.
