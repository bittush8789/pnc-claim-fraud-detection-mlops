import os
import sys
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import time

# Ensure src/ directory is in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from predict import predict_claim
from pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)
app.secret_key = 'super-secret-key-for-pc-insurance-fraud-detection-mlops'

# Metric counters for monitoring
PREDICTION_COUNT = 0
FRAUD_COUNT = 0
TOTAL_LATENCY = 0.0

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

# --- Web UI Routes ---
@app.route('/')
def index():
    stats = get_stats()
    return render_template('index.html', stats=stats)

@app.route('/predict_ui', methods=['GET', 'POST'])
def predict_ui():
    if request.method == 'POST':
        try:
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
            
            res = predict_claim(form_data)
            return render_template('result.html', result=res, inputs=form_data)
            
        except Exception as e:
            flash(f"Error during claim prediction: {str(e)}", "error")
            return redirect(url_for('predict_ui'))
            
    return render_template('predict.html')

# --- MLOps Production APIs ---

@app.route('/health', methods=['GET'])
def health():
    """Kubernetes liveness/readiness probe target."""
    return jsonify({"status": "healthy"}), 200

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus monitoring exposition endpoint."""
    global PREDICTION_COUNT, FRAUD_COUNT, TOTAL_LATENCY
    avg_latency = TOTAL_LATENCY / PREDICTION_COUNT if PREDICTION_COUNT > 0 else 0.0
    metric_text = (
        f"# HELP pnc_predictions_total Total number of processed predictions\n"
        f"# TYPE pnc_predictions_total counter\n"
        f"pnc_predictions_total {PREDICTION_COUNT}\n"
        f"# HELP pnc_fraud_total Total number of fraudulent claims detected\n"
        f"# TYPE pnc_fraud_total counter\n"
        f"pnc_fraud_total {FRAUD_COUNT}\n"
        f"# HELP pnc_avg_latency_seconds Average prediction latency\n"
        f"# TYPE pnc_avg_latency_seconds gauge\n"
        f"pnc_avg_latency_seconds {avg_latency}\n"
    )
    return metric_text, 200, {'Content-Type': 'text/plain; version=0.0.4; charset=utf-8'}

@app.route('/predict', methods=['POST'])
def predict_api():
    """Single claim prediction JSON endpoint."""
    global PREDICTION_COUNT, FRAUD_COUNT, TOTAL_LATENCY
    start_time = time.time()
    try:
        data = request.get_json(force=True)
        res = predict_claim(data)
        
        # Track metrics
        PREDICTION_COUNT += 1
        if res['prediction'] == 1:
            FRAUD_COUNT += 1
        TOTAL_LATENCY += (time.time() - start_time)
        
        return jsonify({
            "prediction": res['prediction_label'],
            "probability": res['probability']
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/batch-predict', methods=['POST'])
def batch_predict_api():
    """Batch claims prediction JSON endpoint."""
    global PREDICTION_COUNT, FRAUD_COUNT, TOTAL_LATENCY
    start_time = time.time()
    try:
        data_list = request.get_json(force=True)
        if not isinstance(data_list, list):
            return jsonify({"error": "Payload must be a list of claims"}), 400
            
        pipeline = PredictionPipeline()
        results = []
        
        for item in data_list:
            pred, prob = pipeline.predict(item)
            PREDICTION_COUNT += 1
            if pred == 1:
                FRAUD_COUNT += 1
            results.append({
                "prediction": "FRAUDULENT CLAIM" if pred == 1 else "LEGITIMATE CLAIM",
                "probability": prob
            })
            
        TOTAL_LATENCY += (time.time() - start_time)
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=True, host='0.0.0.0', port=port)
