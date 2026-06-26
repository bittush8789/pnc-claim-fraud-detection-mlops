import numpy as np
import pandas as pd
import os

def generate_synthetic_data(num_samples=1200, random_state=42):
    np.random.seed(random_state)
    
    # 1. Base categorical choices
    policy_types = ['Auto', 'Home', 'Commercial']
    claim_types = ['Collision', 'Theft', 'Fire']
    incident_types = ['Single Vehicle Collision', 'Multi-Vehicle Collision', 'Theft', 'Fire', 'Water Damage']
    severities = ['Minor', 'Major', 'Total Loss']
    yes_no = ['Yes', 'No']
    states = ['NY', 'CA', 'TX', 'FL', 'IL']
    months = ['January', 'February', 'March', 'April', 'May', 'June', 
              'July', 'August', 'September', 'October', 'November', 'December']
    
    # 2. Draw features
    policy_age_days = np.random.randint(5, 3650, size=num_samples)
    policy_type = np.random.choice(policy_types, size=num_samples, p=[0.5, 0.3, 0.2])
    customer_age = np.random.randint(18, 80, size=num_samples)
    
    # Link claim amount and insured amount
    insured_amount = np.random.choice([10000, 25000, 50000, 100000, 250000, 500000], size=num_samples, p=[0.1, 0.2, 0.3, 0.2, 0.15, 0.05])
    # claim_amount is usually less than or equal to insured_amount, occasionally exceeds it
    claim_ratio = np.random.beta(a=2, b=5, size=num_samples) # centered around 0.3
    # let some be high
    high_claim_indices = np.random.choice(num_samples, size=int(num_samples*0.15), replace=False)
    claim_ratio[high_claim_indices] = np.random.uniform(0.7, 1.1, size=len(high_claim_indices))
    claim_amount = np.round(insured_amount * claim_ratio, 2)
    
    # Policy premium
    premium_amount = np.round(insured_amount * 0.008 + np.random.normal(200, 50, size=num_samples), 2)
    premium_amount = np.clip(premium_amount, 100, 8000)
    
    previous_claims = np.random.choice([0, 1, 2, 3, 4, 5], size=num_samples, p=[0.5, 0.3, 0.12, 0.05, 0.02, 0.01])
    
    claim_type = np.random.choice(claim_types, size=num_samples, p=[0.6, 0.25, 0.15])
    incident_type = []
    for c_t in claim_type:
        if c_t == 'Collision':
            incident_type.append(np.random.choice(['Single Vehicle Collision', 'Multi-Vehicle Collision'], p=[0.4, 0.6]))
        elif c_t == 'Theft':
            incident_type.append('Theft')
        else:
            incident_type.append(np.random.choice(['Fire', 'Water Damage'], p=[0.7, 0.3]))
    incident_type = np.array(incident_type)
    
    incident_severity = np.random.choice(severities, size=num_samples, p=[0.5, 0.35, 0.15])
    witness_present = np.random.choice(yes_no, size=num_samples, p=[0.3, 0.7])
    police_report = np.random.choice(yes_no, size=num_samples, p=[0.6, 0.4])
    claim_submission_delay = np.random.geometric(p=0.2, size=num_samples) - 1 # centered around 4 days
    claim_submission_delay = np.clip(claim_submission_delay, 0, 30)
    
    property_damage = np.random.choice(yes_no, size=num_samples, p=[0.4, 0.6])
    bodily_injuries = np.random.choice([0, 1, 2, 3, 4], size=num_samples, p=[0.6, 0.2, 0.1, 0.07, 0.03])
    vehicles_involved = []
    for idx, inc_t in enumerate(incident_type):
        if 'Multi-Vehicle' in inc_t:
            vehicles_involved.append(np.random.randint(2, 5))
        elif 'Single Vehicle' in inc_t:
            vehicles_involved.append(1)
        else:
            vehicles_involved.append(0)
    vehicles_involved = np.array(vehicles_involved)
    
    state = np.random.choice(states, size=num_samples)
    claim_month = np.random.choice(months, size=num_samples)
    
    # 3. Create logical fraud indicators to build target fraud_reported
    # We will compute a risk score for each row
    risk_score = np.zeros(num_samples)
    
    # Risk factor: Policy age is very low (< 30 days) and claim amount is high
    risk_score += np.where((policy_age_days < 45) & (claim_ratio > 0.6), 2.5, 0)
    
    # Risk factor: Major/Total Loss but no police report
    risk_score += np.where(((incident_severity == 'Major') | (incident_severity == 'Total Loss')) & (police_report == 'No'), 2.0, 0)
    
    # Risk factor: High claim amount but no witness present
    risk_score += np.where((claim_amount > 40000) & (witness_present == 'No'), 1.5, 0)
    
    # Risk factor: High claim to insured ratio
    risk_score += np.where(claim_ratio > 0.85, 2.0, 0)
    
    # Risk factor: Multi-vehicle collision with no bodily injuries but high property damage and high claim
    risk_score += np.where((incident_type == 'Multi-Vehicle Collision') & (bodily_injuries == 0) & (property_damage == 'Yes') & (claim_ratio > 0.7), 1.2, 0)
    
    # Risk factor: High previous claims
    risk_score += np.where(previous_claims >= 3, 1.5, 0)
    
    # Risk factor: Long claim submission delay
    risk_score += np.where(claim_submission_delay > 15, 1.2, 0)
    
    # Add random noise to make it realistic and not 100% separable
    risk_score += np.random.normal(0, 1.5, size=num_samples)
    
    # Pass through sigmoid to get probabilities
    prob = 1 / (1 + np.exp(- (risk_score - 1.5))) # shift to have about 20-25% fraud rate
    
    fraud_reported = np.where(prob > 0.5, 1, 0)
    
    df = pd.DataFrame({
        'policy_age_days': policy_age_days,
        'policy_type': policy_type,
        'customer_age': customer_age,
        'claim_amount': claim_amount,
        'insured_amount': insured_amount,
        'premium_amount': premium_amount,
        'previous_claims': previous_claims,
        'claim_type': claim_type,
        'incident_type': incident_type,
        'incident_severity': incident_severity,
        'witness_present': witness_present,
        'police_report': police_report,
        'claim_submission_delay': claim_submission_delay,
        'property_damage': property_damage,
        'bodily_injuries': bodily_injuries,
        'vehicles_involved': vehicles_involved,
        'state': state,
        'claim_month': claim_month,
        'fraud_reported': fraud_reported
    })
    
    return df

if __name__ == '__main__':
    df = generate_synthetic_data()
    # Save to workspace
    output_path = 'insurance_claims.csv'
    df.to_csv(output_path, index=False)
    print(f"Generated synthetic dataset with {df.shape[0]} rows and {df.shape[1]} columns at {output_path}")
    print(f"Fraud distribution:\n{df['fraud_reported'].value_counts(normalize=True)}")
