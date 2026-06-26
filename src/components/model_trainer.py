import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

class ModelTrainerConfig:
    trained_model_file_path = os.path.join("models", "fraud_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data...")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
                "XGBoost": XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.01, random_state=42, eval_metric='logloss')
            }

            model_report = {}

            import mlflow
            import mlflow.sklearn
            
            # Start MLflow run
            mlflow.set_experiment("PNC_Claim_Fraud_Detection")
            
            for model_name, model in models.items():
                logging.info(f"Training model: {model_name}...")
                
                with mlflow.start_run(run_name=model_name, nested=True):
                    model.fit(X_train, y_train)

                    y_test_pred = model.predict(X_test)
                    y_test_prob = model.predict_proba(X_test)[:, 1]

                    # Evaluate metrics
                    acc = accuracy_score(y_test, y_test_pred)
                    prec = precision_score(y_test, y_test_pred, zero_division=0)
                    rec = recall_score(y_test, y_test_pred, zero_division=0)
                    f1 = f1_score(y_test, y_test_pred, zero_division=0)
                    roc = roc_auc_score(y_test, y_test_prob)

                    # Log hyperparameters and metrics to MLflow
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_metric("accuracy", acc)
                    mlflow.log_metric("precision", prec)
                    mlflow.log_metric("recall", rec)
                    mlflow.log_metric("f1_score", f1)
                    mlflow.log_metric("roc_auc", roc)
                    mlflow.sklearn.log_model(model, artifact_path=model_name)

                    # Custom composite metric emphasizing recall and ROC-AUC
                    composite = 0.6 * rec + 0.4 * roc
                    model_report[model_name] = {
                        "accuracy": acc,
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                        "roc_auc": roc,
                        "composite": composite,
                        "model_obj": model
                    }

                    logging.info(f"{model_name} - Recall: {rec:.4f} | ROC-AUC: {roc:.4f} | F1: {f1:.4f} | Accuracy: {acc:.4f}")

            # Pick best model based on composite score
            best_model_name = max(model_report, key=lambda k: model_report[k]["composite"])
            best_model_data = model_report[best_model_name]
            best_model = best_model_data["model_obj"]

            logging.info(f"Best model selected: {best_model_name} with Composite Score: {best_model_data['composite']:.4f}")

            # Ensure models directory exists
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            
            # Save best model object
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Best model pickled successfully.")

            return best_model_name, best_model_data
        except Exception as e:
            raise CustomException(e, sys)
