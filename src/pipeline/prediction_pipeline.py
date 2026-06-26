import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.components.data_transformation import add_engineered_features

class PredictionPipeline:
    def __init__(self):
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.model_path = os.path.join("models", "fraud_model.pkl")

    def predict(self, input_data):
        logging.info("Initiating prediction pipeline...")
        try:
            # Convert dict or list to DataFrame
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            else:
                df = pd.DataFrame(input_data)

            # Ensure data types match
            df['policy_age_days'] = df['policy_age_days'].astype(int)
            df['customer_age'] = df['customer_age'].astype(int)
            df['claim_amount'] = df['claim_amount'].astype(float)
            df['insured_amount'] = df['insured_amount'].astype(float)
            df['premium_amount'] = df['premium_amount'].astype(float)
            df['previous_claims'] = df['previous_claims'].astype(int)
            df['claim_submission_delay'] = df['claim_submission_delay'].astype(int)
            df['bodily_injuries'] = df['bodily_injuries'].astype(int)
            df['vehicles_involved'] = df['vehicles_involved'].astype(int)

            # 1. Apply feature engineering
            logging.info("Applying feature engineering on input payload...")
            df = add_engineered_features(df)

            # 2. Load preprocessor and model
            logging.info("Loading preprocessor and estimator object...")
            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)

            # 3. Transform inputs
            transformed_features = preprocessor.transform(df)

            # 4. Predict probability and labels
            prob = float(model.predict_proba(transformed_features)[:, 1][0])
            prediction = int(model.predict(transformed_features)[0])

            logging.info(f"Prediction result - Probability: {prob:.4f} | Prediction: {prediction}")
            return prediction, prob
        except Exception as e:
            raise CustomException(e, sys)
