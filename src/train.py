import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score, precision_score, f1_score

from data_ingestion import run_ingestion_pipeline
from feature_engineering import add_engineered_features
from preprocessing import get_preprocessing_pipeline

def train_and_evaluate_models():
    # 1. Ingest data first if not exists
    train_path = 'data/processed/train.csv'
    test_path = 'data/processed/test.csv'
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        train_path, test_path = run_ingestion_pipeline()
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 2. Feature Engineering
    print("Applying feature engineering...")
    train_df = add_engineered_features(train_df)
    test_df = add_engineered_features(test_df)
    
    # Separate features and target
    X_train = train_df.drop(columns=['fraud_reported'])
    y_train = train_df['fraud_reported']
    X_test = test_df.drop(columns=['fraud_reported'])
    y_test = test_df['fraud_reported']
    
    # Get preprocessing
    preprocessor = get_preprocessing_pipeline()
    
    # Define models and their parameter grids for hyperparameter tuning
    models_config = {
        'LogisticRegression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {
                'classifier__C': [0.01, 0.1, 1.0, 10.0]
            }
        },
        'DecisionTree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'classifier__max_depth': [3, 5, 8, None],
                'classifier__min_samples_split': [2, 5, 10]
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [5, 8, 12, None]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, eval_metric='logloss'),
            'params': {
                'classifier__n_estimators': [50, 100, 150],
                'classifier__max_depth': [3, 5, 7],
                'classifier__learning_rate': [0.01, 0.05, 0.1]
            }
        },
        'LightGBM': {
            'model': LGBMClassifier(random_state=42, verbose=-1),
            'params': {
                'classifier__n_estimators': [50, 100, 150],
                'classifier__max_depth': [3, 5, 7],
                'classifier__learning_rate': [0.01, 0.05, 0.1]
            }
        }
    }
    
    best_pipelines = {}
    model_scores = {}
    
    # Train each model using GridSearchCV (optimizing recall/roc_auc)
    for name, config in models_config.items():
        print(f"\n--- Training {name} ---")
        
        # Create full pipeline: preprocessor + model
        clf_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', config['model'])
        ])
        
        # We search with scoring='roc_auc' or 'recall'. Let's use 'roc_auc' for tuning
        grid = GridSearchCV(
            clf_pipeline,
            param_grid=config['params'],
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        best_pipelines[name] = best_model
        
        # Evaluate on training fold / validation metrics
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob)
        
        # Prioritize Recall and ROC-AUC: composite score
        # Since we want to ensure high recall without losing too much precision,
        # let's rank models based on composite score: 0.6 * recall + 0.4 * roc_auc
        composite_score = 0.6 * rec + 0.4 * roc
        
        model_scores[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': roc,
            'composite_score': composite_score
        }
        
        print(f"Best Params: {grid.best_params_}")
        print(f"Recall: {rec:.4f} | ROC-AUC: {roc:.4f} | F1: {f1:.4f} | Accuracy: {acc:.4f} | Precision: {prec:.4f}")
    
    # 3. Select the best model automatically based on composite score
    best_model_name = max(model_scores, key=lambda k: model_scores[k]['composite_score'])
    print(f"\n==========================================")
    print(f"Best model selected: {best_model_name}")
    print(f"Composite Score: {model_scores[best_model_name]['composite_score']:.4f}")
    print(f"==========================================")
    
    # Save best model pipeline
    os.makedirs('models', exist_ok=True)
    model_path = 'models/fraud_model.pkl'
    
    # We want to package feature engineering + preprocessor + estimator into the final pickle
    # However, Python functions like add_engineered_features can be tricky to pickle directly
    # inside standard pipelines. So we will save a dict containing:
    # 1. The name of the best model.
    # 2. The pipeline (preprocessor + classifier).
    # 3. Model score metadata.
    save_data = {
        'model_name': best_model_name,
        'pipeline': best_pipelines[best_model_name],
        'metrics': model_scores[best_model_name]
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(save_data, f)
        
    print(f"Saved best model to {model_path}")
    
    return best_model_name, model_scores

if __name__ == '__main__':
    train_and_evaluate_models()
