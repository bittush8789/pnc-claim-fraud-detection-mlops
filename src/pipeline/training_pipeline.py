import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

class TrainingPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("==========================================")
            logging.info("Starting Training Pipeline execution...")
            logging.info("==========================================")

            # 1. Data Ingestion
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()

            # 2. Data Validation
            validation = DataValidation()
            is_valid = validation.validate_dataset()
            if not is_valid:
                logging.warning("Data Validation failed. Check validation_report.yaml.")
            else:
                logging.info("Data Validation passed.")

            # 3. Data Transformation
            transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(train_path, test_path)

            # 4. Model Training
            trainer = ModelTrainer()
            best_model_name, model_data = trainer.initiate_model_trainer(train_arr, test_arr)

            # 5. Model Evaluation
            evaluation = ModelEvaluation()
            metrics = evaluation.evaluate_model(test_arr)

            logging.info("Training Pipeline execution completed successfully.")
            return metrics
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
