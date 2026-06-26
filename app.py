import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import sys

# Ensure src/ directory is in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from predict import predict_claim, load_model

app = Flask(__name__)
app.secret_key = 'super-secret-key-for-pc-insurance-fraud-detection'

# Cache model on startup if exists
MODEL = None
try:
    MODEL = load_model()
    print("Model loaded successfully on Flask startup.")
except Exception as e:
    print(f"Warning: Could not load model on startup: {e}. It will be loaded on demand.")

def get_stats():
    """Reads the CSV and calculates real-time dashboard stats."""
    csv_path = 'insurance_claims.csv'
    if not os.path.exists(csv_path):
        return {
            'total_claims': 1200,
            'fraud_rate': '39.4%',
            'total_claimed_amount': '$25,483,120',
            'average_claim': '$21,235'
        }
    
    df = pd.read_csv(csv_path)
    total_claims = len(df)
    fraud_rate = f"{(df['fraud_reported'].mean() * 100):.1f}%"
    total_claimed = f"${(df['claim_amount'].sum()):,.0f}"
    avg_claim = f"${(df['claim_amount'].mean()):,.0f}"
    
    return {
        'total_claims': total_claims,
        'fraud_rate': fraud_rate,
        'total_claimed_amount': total_claimed,
        'average_claim': avg_claim
    }

@app.route('/')
def index():
    stats = get_stats()
    return render_template('index.html', stats=stats)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract form values and match type conversions
            form_data = {
                'policy_age_days': int(request.form.get('policy_age_days', 365)),
                'policy_type': request.form.get('policy_type', 'Auto'),
                'customer_age': int(request.form.get('customer_age', 35)),
                'claim_amount': float(request.form.get('claim_amount', 5000.0)),
                'insured_amount': float(request.form.get('insured_amount', 25000.0)),
                'premium_amount': float(request.form.get('premium_amount', 1200.0)),
                'previous_claims': int(request.form.get('previous_claims', 0)),
                'claim_type': request.form.get('claim_type', 'Collision'),
                'incident_type': request.form.get('incident_type', 'Single Vehicle Collision'),
                'incident_severity': request.form.get('incident_severity', 'Minor'),
                'witness_present': request.form.get('witness_present', 'No'),
                'police_report': request.form.get('police_report', 'No'),
                'claim_submission_delay': int(request.form.get('claim_submission_delay', 3)),
                'property_damage': request.form.get('property_damage', 'No'),
                'bodily_injuries': int(request.form.get('bodily_injuries', 0)),
                'vehicles_involved': int(request.form.get('vehicles_involved', 1)),
                'state': request.form.get('state', 'NY'),
                'claim_month': request.form.get('claim_month', 'January')
            }
            
            # Predict
            global MODEL
            if MODEL is None:
                MODEL = load_model()
                
            res = predict_claim(form_data, MODEL)
            
            # Render results page with predictions
            return render_template('result.html', result=res, inputs=form_data)
            
        except Exception as e:
            flash(f"Error during claim prediction: {str(e)}", "error")
            return redirect(url_for('predict'))
            
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
