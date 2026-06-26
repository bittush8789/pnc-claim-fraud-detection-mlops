import os
import sys
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class ModelEvaluation:
    def __init__(self):
        self.metrics_file_path = "metrics.json"

    def evaluate_model(self, test_array, model_path="models/fraud_model.pkl"):
        logging.info("Initiating model evaluation pipeline...")
        try:
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            # Load model
            model = load_object(model_path)
            
            # Run predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, y_prob))
            }

            with open(self.metrics_file_path, "w") as f:
                json.dump(metrics, f, indent=4)
            logging.info(f"Model evaluation complete. Metrics written to {self.metrics_file_path}")
            return metrics
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    pass
