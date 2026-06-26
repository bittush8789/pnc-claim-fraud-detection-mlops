import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import write_yaml

REQUIRED_COLUMNS = {
    'policy_age_days': 'int',
    'policy_type': 'object',
    'customer_age': 'int',
    'claim_amount': 'float',
    'insured_amount': 'int',
    'premium_amount': 'float',
    'previous_claims': 'int',
    'claim_type': 'object',
    'incident_type': 'object',
    'incident_severity': 'object',
    'witness_present': 'object',
    'police_report': 'object',
    'claim_submission_delay': 'int',
    'property_damage': 'object',
    'bodily_injuries': 'int',
    'vehicles_involved': 'int',
    'state': 'object',
    'claim_month': 'object',
    'fraud_reported': 'int'
}

class DataValidation:
    def __init__(self):
        self.validation_report_path = "validation_report.yaml"

    def validate_dataset(self, file_path="artifacts/raw.csv"):
        logging.info(f"Initiating data validation on {file_path}...")
        try:
            df = pd.read_csv(file_path)
            validation_status = True
            errors = []

            # 1. Column existence & datatype validation
            for col, expected_type in REQUIRED_COLUMNS.items():
                if col not in df.columns:
                    validation_status = False
                    errors.append(f"Missing column: {col}")
                    
            # 2. Check duplicate count
            duplicate_count = int(df.duplicated().sum())

            # 3. Check missing count
            missing_count = int(df.isnull().sum().sum())

            report = {
                "validation_status": validation_status,
                "dataset_shape": list(df.shape),
                "duplicate_count": duplicate_count,
                "missing_count": missing_count,
                "errors": errors
            }

            write_yaml(self.validation_report_path, report)
            logging.info(f"Validation report saved to {self.validation_report_path}")
            return validation_status
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    val = DataValidation()
    val.validate_dataset()
