import sys
from src.pipeline.training_pipeline import TrainingPipeline
from src.logger import logging
from src.exception import CustomException

def main():
    try:
        pipeline = TrainingPipeline()
        metrics = pipeline.run_pipeline()
        print("Training pipeline ran successfully.")
        print("Metrics:")
        print(metrics)
    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)

if __name__ == '__main__':
    main()
