
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import json
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score

# Set project root
project_root = Path(os.getcwd())
sys.path.insert(0, str(project_root))

from models.architectures.symptom_classifier import SymptomCategoryClassifier, SymptomDiseaseClassifier

def calculate_top_k():
    print("Loading resources...")
    checkpoint_dir = project_root / "models" / "checkpoints"
    
    # Load encoders
    disease_encoder = joblib.load(checkpoint_dir / "disease_encoder.pkl")
    category_encoder = joblib.load(checkpoint_dir / "category_encoder.pkl")
    
    # Load data
    print("Loading data...")
    df_demo = pd.read_csv(project_root / "data" / "processed" / "symptoms" / "symptoms_augmented_with_demographics.csv")
    
    # Filter valid diseases
    valid_diseases = set(disease_encoder.classes_)
    df_demo = df_demo[df_demo['diseases'].isin(valid_diseases)].reset_index(drop=True)
    
    # Prepare features
    non_feature_cols = ['diseases', 'disease_category', 'symptoms', 'age', 'sex']
    feature_cols = [c for c in df_demo.columns if c not in non_feature_cols]
    
    df_demo['sex_encoded'] = (df_demo['sex'] == 'M').astype(int)
    df_demo['age_normalized'] = df_demo['age'] / 100.0
    
    X_demo = df_demo[feature_cols + ['age_normalized', 'sex_encoded']].values
    y_cat_demo = category_encoder.transform(df_demo['disease_category'].values)
    y_dis_demo = disease_encoder.transform(df_demo['diseases'].values)
    
    # Split data
    print("Splitting data...")
    _, X_test_d, _, y_cat_test_d, _, y_dis_test_d = train_test_split(
        X_demo, y_cat_demo, y_dis_demo, test_size=0.1, random_state=42, stratify=y_cat_demo
    )
    
    # Load models
    print("Loading classifiers...")
    cat_clf_demo = SymptomCategoryClassifier.load(str(checkpoint_dir / "category_classifier_demographics.pkl"))
    dis_clf_demo = SymptomDiseaseClassifier.load(str(checkpoint_dir / "disease_classifier_demographics.pkl"))
    
    # Predict probabilities
    print("Predicting probabilities...")
    cat_proba_d = cat_clf_demo.predict_proba(X_test_d)
    dis_proba_d = dis_clf_demo.predict_proba(X_test_d)
    
    # Calculate Top-k
    print("\n--- Category Classification (+Demo) ---")
    cat_labels = np.arange(len(category_encoder.classes_))
    for k in range(1, 11):
        if k <= len(cat_labels):
            acc = top_k_accuracy_score(y_cat_test_d, cat_proba_d, k=k, labels=cat_labels)
            print(f"Top-{k}: {acc*100:.2f}%")
        else:
            print(f"Top-{k}: 100.00% (All {len(cat_labels)} classes covered)")
            
    print("\n--- Disease Classification (+Demo) ---")
    dis_labels = np.arange(len(disease_encoder.classes_))
    for k in range(1, 11):
        acc = top_k_accuracy_score(y_dis_test_d, dis_proba_d, k=k, labels=dis_labels)
        print(f"Top-{k}: {acc*100:.2f}%")

if __name__ == "__main__":
    calculate_top_k()
