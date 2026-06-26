import kfp
from kfp import dsl
from kfp.dsl import component, Output, Artifact, Input, Dataset, Model, Metrics

@component(base_image="python:3.11-slim")
def data_ingestion(
    raw_dataset: Output[Dataset],
    train_split: Output[Dataset],
    test_split: Output[Dataset]
):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # Load dataset
    df = pd.read_csv("https://raw.githubusercontent.com/bittush8789/pnc-claim-fraud-detection-mlops/cicd/insurance_claims.csv")
    df.to_csv(raw_dataset.path, index=False)
    
    # Train test split
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df['fraud_reported'])
    train_set.to_csv(train_split.path, index=False)
    test_set.to_csv(test_split.path, index=False)

@component(base_image="python:3.11-slim")
def data_validation(
    dataset: Input[Dataset],
    validation_report: Output[Artifact]
):
    import pandas as pd
    import yaml
    
    df = pd.read_csv(dataset.path)
    required_cols = [
        'policy_age_days', 'policy_type', 'customer_age', 'claim_amount',
        'insured_amount', 'premium_amount', 'previous_claims', 'claim_type',
        'incident_type', 'incident_severity', 'witness_present', 'police_report',
        'claim_submission_delay', 'property_damage', 'bodily_injuries',
        'vehicles_involved', 'state', 'claim_month', 'fraud_reported'
    ]
    
    validation_status = True
    errors = []
    
    for col in required_cols:
        if col not in df.columns:
            validation_status = False
            errors.append(f"Missing column: {col}")
            
    report = {
        "validation_status": validation_status,
        "dataset_shape": list(df.shape),
        "errors": errors
    }
    
    with open(validation_report.path, 'w') as f:
        yaml.safe_dump(report, f)

@component(base_image="python:3.11-slim")
def data_transformation(
    train_split: Input[Dataset],
    test_split: Input[Dataset],
    preprocessor_artifact: Output[Artifact],
    transformed_train: Output[Dataset],
    transformed_test: Output[Dataset]
):
    import pandas as pd
    import numpy as np
    import pickle
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    
    train_df = pd.read_csv(train_split.path)
    test_df = pd.read_csv(test_split.path)
    
    # Custom Feature Engineering
    for df in [train_df, test_df]:
        df['claim_to_insured_ratio'] = df['claim_amount'] / (df['insured_amount'] + 1e-5)
        df['claim_to_premium_ratio'] = df['claim_amount'] / (df['premium_amount'] + 1e-5)
        policy_years = df['policy_age_days'] / 365.25
        df['previous_claim_frequency'] = df['previous_claims'] / (policy_years + 0.1)
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
    
    target_col = "fraud_reported"
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    # Save transformed arrays
    np.save(transformed_train.path, np.c_[X_train_trans, y_train])
    np.save(transformed_test.path, np.c_[X_test_trans, y_test])
    
    with open(preprocessor_artifact.path, 'wb') as f:
        pickle.dump(preprocessor, f)

@component(base_image="python:3.11-slim")
def model_training(
    train_dataset: Input[Dataset],
    test_dataset: Input[Dataset],
    model_artifact: Output[Model],
    performance_metrics: Output[Metrics]
):
    import numpy as np
    import pickle
    import mlflow
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    train_arr = np.load(train_dataset.path + ".npy")
    test_arr = np.load(test_dataset.path + ".npy")
    
    X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
    X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
    
    # Start MLflow Tracking
    mlflow.set_tracking_uri("http://localhost:5000") # local tracking fallback
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob)
        
        # Log to MLflow
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("roc_auc", roc)
        
        mlflow.sklearn.log_model(model, "model")
        
        # Write to Kubeflow output metrics
        performance_metrics.log_metric("accuracy", float(acc))
        performance_metrics.log_metric("precision", float(prec))
        performance_metrics.log_metric("recall", float(rec))
        performance_metrics.log_metric("roc_auc", float(roc))
        
        with open(model_artifact.path, 'wb') as f:
            pickle.dump(model, f)

@dsl.pipeline(
    name="pc-claims-fraud-detection",
    description="End-to-End P&C Insurance Fraud Detection Training Pipeline"
)
def claims_pipeline():
    ingest = data_ingestion()
    
    validate = data_validation(
        dataset=ingest.outputs['raw_dataset']
    )
    
    transform = data_transformation(
        train_split=ingest.outputs['train_split'],
        test_split=ingest.outputs['test_split']
    )
    
    train = model_training(
        train_dataset=transform.outputs['transformed_train'],
        test_dataset=transform.outputs['transformed_test']
    )

if __name__ == '__main__':
    # Compiles pipeline definition to YAML spec
    import kfp.compiler as compiler
    compiler.Compiler().compile(
        pipeline_func=claims_pipeline,
        package_path='claims_pipeline.yaml'
    )
    print("Kubeflow Pipeline compiled to claims_pipeline.yaml successfully.")
