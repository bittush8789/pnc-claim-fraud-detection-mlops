import os
import pickle
import pandas as pd
import numpy as np

# Import custom feature engineering
from feature_engineering import add_engineered_features

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'fraud_model.pkl')

def load_model():
    """
    Loads and returns the trained model pipeline.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at: {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as f:
        saved_data = pickle.load(f)
    return saved_data['pipeline']

def preprocess_input(input_data):
    """
    Converts raw dictionary input data into a DataFrame and applies feature engineering.
    """
    # Convert single dictionary to DataFrame
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df = pd.DataFrame(input_data)
    else:
        df = input_data.copy()
        
    # Standardize column data types to match expected model features
    df['policy_age_days'] = df['policy_age_days'].astype(int)
    df['customer_age'] = df['customer_age'].astype(int)
    df['claim_amount'] = df['claim_amount'].astype(float)
    df['insured_amount'] = df['insured_amount'].astype(float)
    df['premium_amount'] = df['premium_amount'].astype(float)
    df['previous_claims'] = df['previous_claims'].astype(int)
    df['claim_submission_delay'] = df['claim_submission_delay'].astype(int)
    df['bodily_injuries'] = df['bodily_injuries'].astype(int)
    df['vehicles_involved'] = df['vehicles_involved'].astype(int)
    
    # Run feature engineering
    df = add_engineered_features(df)
    return df

def calculate_probability(input_data, model=None):
    """
    Predicts the probability of fraud for the input data.
    """
    if model is None:
        model = load_model()
    
    # Preprocess
    X_processed = preprocess_input(input_data)
    
    # Predict probabilities
    probs = model.predict_proba(X_processed)[:, 1]
    return float(probs[0]) if len(probs) == 1 else probs

def assign_risk_level(probability):
    """
    Assigns risk level and recommendation based on the fraud probability.
    Rules:
      0-40%     -> LOW RISK
      41-70%    -> MEDIUM RISK
      71-100%   -> HIGH RISK
    """
    prob_percentage = int(probability * 100)
    
    if prob_percentage <= 40:
        return {
            'level': 'LOW',
            'percentage': prob_percentage,
            'recommendation': 'Approve Claim (Standard Processing)',
            'color': '#10b981' # Green
        }
    elif prob_percentage <= 70:
        return {
            'level': 'MEDIUM',
            'percentage': prob_percentage,
            'recommendation': 'Further Review Recommended',
            'color': '#f59e0b' # Amber/Orange
        }
    else:
        return {
            'level': 'HIGH',
            'percentage': prob_percentage,
            'recommendation': 'Manual Investigation Required (Route to SIU)',
            'color': '#ef4444' # Red
        }

def generate_ai_narrative(inputs, result):
    """
    Generates a realistic Generative AI Co-Pilot analysis report based on claim parameters.
    """
    risk_pct = result['risk_percentage']
    risk_lvl = result['risk_level']
    verdict = result['prediction_label']
    
    # 1. Executive Summary
    summary = f"ShieldGuard AI Claim Copilot has analyzed this claim. Based on our ensemble classification model, the claim carries a **{risk_lvl} RISK** profile with a fraud probability of **{risk_pct}%**. The automated verdict is **{verdict}**."
    
    # 2. Key Flags
    flags = []
    if inputs['policy_age_days'] < 90:
        flags.append(f"**Short Policy Age**: The claim was filed just {inputs['policy_age_days']} days after policy inception, indicating high risk of premium staging.")
    if (inputs['incident_severity'] in ['Major', 'Total Loss']) and inputs['police_report'] == 'No':
        flags.append(f"**Unreported Severe Loss**: The loss is classified as severe ({inputs['incident_severity']}), but no official police report was filed.")
    if inputs['claim_amount'] > (inputs['insured_amount'] * 0.85):
        ratio = (inputs['claim_amount'] / inputs['insured_amount']) * 100
        flags.append(f"**High Loss Ratio**: The claimed amount is {ratio:.1f}% of the maximum policy limit, suggesting potential exaggeration of damages.")
    if inputs['previous_claims'] >= 3:
        flags.append(f"**Prior Frequency**: The claimant has a history of {inputs['previous_claims']} previous claims, which is statistically higher than standard baseline distributions.")
    if inputs['claim_submission_delay'] > 15:
        flags.append(f"**Extended Submission Delay**: A {inputs['claim_submission_delay']}-day delay in reporting the incident is considered atypical for standard collision reports.")
    if inputs['witness_present'] == 'No' and inputs['claim_amount'] > 20000:
        flags.append("**High Value Unwitnessed Incident**: The incident occurred without third-party witness corroboration despite a substantial claimed loss.")
        
    if not flags:
        flags.append("No abnormal or high-risk demographic flags detected. Numerical metrics fall within normal claim limits.")
        
    # 3. Investigation Plan
    plan = []
    if risk_lvl == 'HIGH':
        plan = [
            "**Physical Verification**: Dispatch an inspector to verify the vehicle/property location and check for pre-existing wear pattern mismatches.",
            "**Verbal Statements**: Conduct recorded statements with the claimant and cross-reference timestamps of phone calls.",
            "**Police/Public Records check**: Reach out to regional traffic/police departments to confirm if a dispatch record was created on the incident date.",
            "**Social Media Screening**: Check claimant public posts matching the event day to confirm location and timeline validity."
        ]
    elif risk_lvl == 'MEDIUM':
        plan = [
            "**Receipt Verification**: Request original, itemized parts receipts and shop labor logs to prevent billing inflation.",
            "**Adjuster Verification**: Double check claim photos against local collision damage databases to check for duplicate image reuse."
        ]
    else:
        plan = [
            "**Standard Straight-Through Processing**: No special investigative steps recommended. The claim is cleared for normal settlement."
        ]
        
    return {
        'summary': summary,
        'flags': flags,
        'plan': plan
    }

def predict_claim(input_data, model=None):
    """
    Predicts fraud status (0 or 1), probability, and assigns risk level.
    """
    if model is None:
        model = load_model()
        
    prob = calculate_probability(input_data, model)
    risk_info = assign_risk_level(prob)
    
    # Binary prediction based on standard threshold of 0.5
    prediction = 1 if prob >= 0.5 else 0
    
    res = {
        'prediction': prediction,
        'prediction_label': 'FRAUDULENT CLAIM' if prediction == 1 else 'LEGITIMATE CLAIM',
        'probability': prob,
        'risk_level': risk_info['level'],
        'risk_percentage': risk_info['percentage'],
        'recommendation': risk_info['recommendation'],
        'color': risk_info['color']
    }
    
    # Generate AI Narrative Copilot feature
    res['ai_narrative'] = generate_ai_narrative(input_data, res)
    return res

if __name__ == '__main__':
    # Dry run prediction test
    sample_input = {
        'policy_age_days': 120,
        'policy_type': 'Auto',
        'customer_age': 35,
        'claim_amount': 15000,
        'insured_amount': 50000,
        'premium_amount': 800,
        'previous_claims': 0,
        'claim_type': 'Collision',
        'incident_type': 'Single Vehicle Collision',
        'incident_severity': 'Minor',
        'witness_present': 'Yes',
        'police_report': 'Yes',
        'claim_submission_delay': 2,
        'property_damage': 'No',
        'bodily_injuries': 0,
        'vehicles_involved': 1,
        'state': 'CA',
        'claim_month': 'March'
    }
    
    try:
        res = predict_claim(sample_input)
        print("Dry run result:", res)
    except Exception as e:
        print("Prediction dry run error (make sure model is trained):", e)
