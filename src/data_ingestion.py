import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = [
    'policy_age_days', 'policy_type', 'customer_age', 'claim_amount',
    'insured_amount', 'premium_amount', 'previous_claims', 'claim_type',
    'incident_type', 'incident_severity', 'witness_present', 'police_report',
    'claim_submission_delay', 'property_damage', 'bodily_injuries',
    'vehicles_involved', 'state', 'claim_month', 'fraud_reported'
]

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    return pd.read_csv(file_path)

def validate_schema(df):
    print("Validating schema...")
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema validation failed. Missing columns: {missing_cols}")
    print("Schema is valid.")
    return True

def clean_data(df):
    print("Cleaning data...")
    # Drop duplicates
    initial_shape = df.shape
    df = df.drop_duplicates()
    if df.shape != initial_shape:
        print(f"Dropped {initial_shape[0] - df.shape[0]} duplicate rows.")
    
    # Check and handle missing values
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print("Found missing values:")
        print(missing_counts[missing_counts > 0])
        # We will handle numerical missing values during preprocessing with imputer
        # but here we can just log it
    else:
        print("No missing values found.")
        
    return df

def save_splits(df, raw_dir, processed_dir, test_size=0.2, random_state=42):
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save full cleaned data to raw directory
    raw_path = os.path.join(raw_dir, 'insurance_claims_raw.csv')
    df.to_csv(raw_path, index=False)
    print(f"Saved raw data to {raw_path}")
    
    # Train-test split
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['fraud_reported'])
    
    # Save splits
    train_path = os.path.join(processed_dir, 'train.csv')
    test_path = os.path.join(processed_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved train set to {train_path} ({train_df.shape[0]} rows)")
    print(f"Saved test set to {test_path} ({test_df.shape[0]} rows)")
    
    return train_path, test_path

def run_ingestion_pipeline(file_path='insurance_claims.csv', raw_dir='data/raw', processed_dir='data/processed'):
    df = load_data(file_path)
    validate_schema(df)
    df = clean_data(df)
    train_path, test_path = save_splits(df, raw_dir, processed_dir)
    print("Data ingestion pipeline completed successfully.")
    return train_path, test_path

if __name__ == '__main__':
    run_ingestion_pipeline()
