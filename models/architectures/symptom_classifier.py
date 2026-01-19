import lightgbm as lgb
import numpy as np
from typing import List, Dict, Optional, Tuple
import joblib 
import json

class SymptomCategoryClassifier:
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.05, 
                max_depth: int = 5, class_weight: str = 'balanced', stopping_rounds: int = 10):
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=31,
            class_weight=class_weight,
            random_state=42,
            verbose=-1,
            n_jobs=4
        )
        self.categories = None
        self.feature_names = None
        self.stopping_rounds = stopping_rounds    

    def fit(self, X, y, eval_set= None):
        self.categories = sorted(np.unique(y))
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        
        print(f"Training {self.__class__.__name__} with shape {X.shape}")
        
        if eval_set is not None:
            self.model.fit(
                X, y,
                eval_set=eval_set,
                eval_metric='multi_logloss',
                callbacks=[lgb.early_stopping(stopping_rounds=self.stopping_rounds, verbose=True)]
            )
        else: 
            self.model.fit(X,y)
        return self
    
    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, top_n: int = 20) -> List[Tuple[str, float]]:
        if self.feature_names is None: 
            return []
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[-top_n:][::-1]
        return [(self.feature_names[i], importances[i]) for i in indices]
    
    def save(self, path: str):
        model_data = {
            'model': self.model,
            'categories': self.categories,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, path)
    
    @classmethod
    def load(cls, path: str):
        model_data = joblib.load(path)
        instance = cls()
        instance.model = model_data['model']
        instance.categories = model_data['categories']
        instance.feature_names = model_data['feature_names']
        
        return instance 

class SymptomDiseaseClassifier:
    def __init__(self, category: str,  n_estimators: int = 200, 
                 learning_rate: float = 0.05, max_depth: int = 5, 
                 class_weight: Optional[Dict] = None, stopping_rounds: int = 10):
        self.category = category
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=31,
            class_weight=class_weight if class_weight else 'balanced',
            random_state=42,
            verbose=-1,
            n_jobs=4
        )
        self.diseases = None
        self.feature_names = None
        self.stopping_rounds = stopping_rounds
    
    def fit(self, X, y, eval_set = None):
        self.diseases = sorted(np.unique(y))
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        
        print(f"Training {self.__class__.__name__} for category '{self.category}' with shape {X.shape}")

        if eval_set is not None:
            self.model.fit(
                X, y,
                eval_set= eval_set,
                eval_metric='multi_logloss',
                callbacks=[lgb.early_stopping(stopping_rounds=self.stopping_rounds, verbose=True)]
            )
        else:
            self.model.fit(X,y)
        
        return self
    
    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def save(self, path: str):
        model_data = {
            'model': self.model,
            'category': self.category,
            'diseases': self.diseases,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, path)
    
    @classmethod
    def load(cls, path: str):
        model_data = joblib.load(path)
        instance = cls(category=model_data['category'])
        instance.model = model_data['model']
        instance.diseases = model_data['diseases']
        instance.feature_names = model_data['feature_names']
        
        return instance

    