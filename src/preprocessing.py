from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Define feature columns
NUMERICAL_FEATURES = [
    'policy_age_days', 'customer_age', 'claim_amount', 'insured_amount',
    'premium_amount', 'previous_claims', 'claim_submission_delay',
    'bodily_injuries', 'vehicles_involved',
    # Engineered numeric features
    'claim_to_insured_ratio', 'claim_to_premium_ratio', 'previous_claim_frequency',
    'delay_risk_score', 'customer_risk_score', 'incident_risk_score'
]

CATEGORICAL_FEATURES = [
    'policy_type', 'claim_type', 'incident_type', 'incident_severity',
    'witness_present', 'police_report', 'property_damage', 'state', 'claim_month'
]

def get_preprocessing_pipeline():
    """
    Creates and returns the preprocessor (ColumnTransformer) pipeline.
    """
    # Pipeline for numerical features
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline for categorical features
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine numerical and categorical pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERICAL_FEATURES),
            ('cat', cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    
    return preprocessor
