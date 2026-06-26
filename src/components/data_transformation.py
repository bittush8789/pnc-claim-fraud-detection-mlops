import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# We define the engineered features here directly for encapsulation
def add_engineered_features(df):
    df = df.copy()
    df['claim_to_insured_ratio'] = df['claim_amount'] / (df['insured_amount'] + 1e-5)
    df['claim_to_premium_ratio'] = df['claim_amount'] / (df['premium_amount'] + 1e-5)
    policy_years = df['policy_age_days'] / 365.25
    df['previous_claim_frequency'] = df['previous_claims'] / (policy_years + 0.1)
    
    # Risk scores
    df['delay_risk_score'] = np.where(df['claim_submission_delay'] > 15, 2.0, 
                                      np.where(df['claim_submission_delay'] == 0, 1.0, 0.0))
    
    cond_young = df['customer_age'] < 25
    cond_new_policy = df['policy_age_days'] < 90
    df['customer_risk_score'] = np.where(cond_young & cond_new_policy, 3.0,
                                         np.where(cond_young, 1.5,
                                                  np.where(cond_new_policy, 1.0, 0.0)))
    
    cond_severe = (df['incident_severity'] == 'Major') | (df['incident_severity'] == 'Total Loss')
    df['incident_risk_score'] = np.where(cond_severe & (df['police_report'] == 'No'), 2.0, 0.0)
    df['incident_risk_score'] += np.where(cond_severe & (df['witness_present'] == 'No'), 1.5, 0.0)
    df['incident_risk_score'] += np.where(df['property_damage'] == 'Yes', 0.5, 0.0)
    
    return df

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            num_features = [
                'policy_age_days', 'customer_age', 'claim_amount', 'insured_amount',
                'premium_amount', 'previous_claims', 'claim_submission_delay',
                'bodily_injuries', 'vehicles_involved',
                'claim_to_insured_ratio', 'claim_to_premium_ratio', 'previous_claim_frequency',
                'delay_risk_score', 'customer_risk_score', 'incident_risk_score'
            ]
            cat_features = [
                'policy_type', 'claim_type', 'incident_type', 'incident_severity',
                'witness_present', 'police_report', 'property_damage', 'state', 'claim_month'
            ]

            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            preprocessor = ColumnTransformer([
                ('num', num_pipeline, num_features),
                ('cat', cat_pipeline, cat_features)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Applying custom feature engineering inside data transformation pipeline...")
            train_df = add_engineered_features(train_df)
            test_df = add_engineered_features(test_df)

            target_column_name = "fraud_reported"
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            preprocessing_obj = self.get_data_transformer_object()
            
            logging.info("Fitting transformer and transforming splits...")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
