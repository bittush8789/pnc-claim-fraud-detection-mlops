import pandas as pd
import numpy as np

def add_engineered_features(df):
    """
    Adds engineered features to the claims dataframe.
    This function must handle both single rows (dict/DataFrame) and full datasets.
    """
    # Create a copy to prevent modifying the original
    df = df.copy()
    
    # 1. claim_to_insured_ratio
    df['claim_to_insured_ratio'] = df['claim_amount'] / (df['insured_amount'] + 1e-5)
    
    # 2. claim_to_premium_ratio
    df['claim_to_premium_ratio'] = df['claim_amount'] / (df['premium_amount'] + 1e-5)
    
    # 3. previous_claim_frequency (claims per year of policy age)
    policy_years = df['policy_age_days'] / 365.25
    df['previous_claim_frequency'] = df['previous_claims'] / (policy_years + 0.1)
    
    # 4. delay_risk_score
    # Higher delay can be suspicious, or 0 delay (immediate claim submission)
    df['delay_risk_score'] = 0.0
    if isinstance(df['claim_submission_delay'], pd.Series):
        df.loc[df['claim_submission_delay'] > 15, 'delay_risk_score'] = 2.0
        df.loc[df['claim_submission_delay'] == 0, 'delay_risk_score'] = 1.0
    else:
        # Scalar fallback (for single instance prediction)
        if df['claim_submission_delay'] > 15:
            df['delay_risk_score'] = 2.0
        elif df['claim_submission_delay'] == 0:
            df['delay_risk_score'] = 1.0
            
    # 5. customer_risk_score
    # High risk if customer is young (< 25) or policy is very new (< 60 days)
    df['customer_risk_score'] = 0.0
    
    cond_young = df['customer_age'] < 25
    cond_new_policy = df['policy_age_days'] < 90
    
    if isinstance(df['customer_age'], pd.Series):
        df.loc[cond_young & cond_new_policy, 'customer_risk_score'] = 3.0
        df.loc[cond_young & ~cond_new_policy, 'customer_risk_score'] = 1.5
        df.loc[~cond_young & cond_new_policy, 'customer_risk_score'] = 1.0
    else:
        if cond_young and cond_new_policy:
            df['customer_risk_score'] = 3.0
        elif cond_young:
            df['customer_risk_score'] = 1.5
        elif cond_new_policy:
            df['customer_risk_score'] = 1.0

    # 6. incident_risk_score
    # High risk for major accidents without police reports, or total loss with no witnesses
    df['incident_risk_score'] = 0.0
    
    cond_severe = (df['incident_severity'] == 'Major') | (df['incident_severity'] == 'Total Loss')
    cond_no_police = df['police_report'] == 'No'
    cond_no_witness = df['witness_present'] == 'No'
    
    if isinstance(df['incident_severity'], pd.Series):
        df.loc[cond_severe & cond_no_police, 'incident_risk_score'] += 2.0
        df.loc[cond_severe & cond_no_witness, 'incident_risk_score'] += 1.5
        df.loc[df['property_damage'] == 'Yes', 'incident_risk_score'] += 0.5
    else:
        if cond_severe and cond_no_police:
            df['incident_risk_score'] += 2.0
        if cond_severe and cond_no_witness:
            df['incident_risk_score'] += 1.5
        if df['property_damage'] == 'Yes':
            df['incident_risk_score'] += 0.5
            
    return df

if __name__ == '__main__':
    # Test feature engineering
    test_df = pd.DataFrame([{
        'policy_age_days': 30,
        'policy_type': 'Auto',
        'customer_age': 22,
        'claim_amount': 8000,
        'insured_amount': 10000,
        'premium_amount': 500,
        'previous_claims': 1,
        'claim_type': 'Collision',
        'incident_type': 'Single Vehicle Collision',
        'incident_severity': 'Major',
        'witness_present': 'No',
        'police_report': 'No',
        'claim_submission_delay': 0,
        'property_damage': 'Yes',
        'bodily_injuries': 0,
        'vehicles_involved': 1,
        'state': 'NY',
        'claim_month': 'January'
    }])
    engineered = add_engineered_features(test_df)
    print("Engineered Features columns:", list(engineered.columns))
    print("Computed risk scores:")
    print(engineered[['claim_to_insured_ratio', 'claim_to_premium_ratio', 'previous_claim_frequency', 
                      'delay_risk_score', 'customer_risk_score', 'incident_risk_score']].to_dict(orient='records')[0])
