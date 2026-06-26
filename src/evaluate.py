import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from feature_engineering import add_engineered_features

# Set plot style for premium dark UI
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    'figure.facecolor': '#0f172a', # slate 900
    'axes.facecolor': '#1e293b',   # slate 800
    'text.color': '#f8fafc',       # slate 50
    'axes.labelcolor': '#cbd5e1',  # slate 300
    'xtick.color': '#94a3b8',      # slate 400
    'ytick.color': '#94a3b8',      # slate 400
    'font.family': 'sans-serif',
    'axes.edgecolor': '#334155'
})

def generate_eda_plots(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print("Generating Exploratory Data Analysis (EDA) plots...")
    
    # 1. Fraud distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='fraud_reported', data=df, hue='fraud_reported', palette=['#10b981', '#ef4444'], legend=False)
    plt.title('Fraud Distribution', fontsize=14, color='#f8fafc', weight='bold')
    plt.xlabel('Fraud Reported (0=No, 1=Yes)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fraud_distribution.png'), dpi=150, facecolor='#0f172a')
    plt.close()
    
    # 2. Claim amount distribution by fraud
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x='claim_amount', hue='fraud_reported', kde=True, palette=['#10b981', '#ef4444'], bins=30, alpha=0.6)
    plt.title('Claim Amount Distribution by Fraud Status', fontsize=14, color='#f8fafc', weight='bold')
    plt.xlabel('Claim Amount ($)')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'claim_amount_distribution.png'), dpi=150, facecolor='#0f172a')
    plt.close()
    
    # 3. Policy type analysis
    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x='policy_type', hue='fraud_reported', palette=['#10b981', '#ef4444'])
    plt.title('Fraud rate by Policy Type', fontsize=14, color='#f8fafc', weight='bold')
    plt.xlabel('Policy Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'policy_type_analysis.png'), dpi=150, facecolor='#0f172a')
    plt.close()
    
    # 4. Incident severity analysis
    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x='incident_severity', hue='fraud_reported', order=['Minor', 'Major', 'Total Loss'], palette=['#10b981', '#ef4444'])
    plt.title('Incident Severity Analysis', fontsize=14, color='#f8fafc', weight='bold')
    plt.xlabel('Incident Severity')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'severity_analysis.png'), dpi=150, facecolor='#0f172a')
    plt.close()
    
    # 5. Outlier analysis
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x='fraud_reported', y='claim_amount', hue='fraud_reported', palette=['#10b981', '#ef4444'], legend=False)
    plt.title('Outlier Analysis: Claim Amount Boxplot', fontsize=14, color='#f8fafc', weight='bold')
    plt.xlabel('Fraud Reported')
    plt.ylabel('Claim Amount ($)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'outlier_analysis.png'), dpi=150, facecolor='#0f172a')
    plt.close()
    
    # 6. Fraud trends by state
    plt.figure(figsize=(7, 4))
    state_fraud = df.groupby('state')['fraud_reported'].mean().reset_index()
    sns.barplot(data=state_fraud, x='state', y='fraud_reported', palette='crest')
    plt.title('Fraud Rate by State', fontsize=14, color='#f8fafc', weight='bold')
    plt.xlabel('State')
    plt.ylabel('Fraud Rate (Mean)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'state_trends.png'), dpi=150, facecolor='#0f172a')
    plt.close()
    
    # 7. Fraud trends by claim type
    plt.figure(figsize=(7, 4))
    sns.barplot(data=df, x='claim_type', y='fraud_reported', hue='claim_type', palette='viridis', legend=False)
    plt.title('Fraud Rate by Claim Type', fontsize=14, color='#f8fafc', weight='bold')
    plt.xlabel('Claim Type')
    plt.ylabel('Fraud Rate (Mean)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'claim_type_trends.png'), dpi=150, facecolor='#0f172a')
    plt.close()

    # 8. Correlation heatmap (numeric columns only)
    plt.figure(figsize=(10, 8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Filter out columns with no variance
    numeric_cols = [c for c in numeric_cols if df[c].std() > 0]
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5, cbar=True)
    plt.title('Correlation Heatmap', fontsize=14, color='#f8fafc', weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=150, facecolor='#0f172a')
    plt.close()

def generate_evaluation_plots(y_test, y_pred, y_prob, output_dir, pipeline, feature_names):
    os.makedirs(output_dir, exist_ok=True)
    print("Generating evaluation plots...")
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Legitimate', 'Fraudulent'],
                yticklabels=['Legitimate', 'Fraudulent'])
    plt.title('Confusion Matrix', fontsize=14, color='#f8fafc', weight='bold')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150, facecolor='#0f172a')
    plt.close()
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='#8b5cf6', lw=3, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#64748b', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=14, color='#f8fafc', weight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=150, facecolor='#0f172a')
    plt.close()
    
    # 3. Feature Importance
    try:
        # Get estimator and preprocessor
        preprocessor = pipeline.named_steps['preprocessor']
        classifier = pipeline.named_steps['classifier']
        
        # Get transformed feature names
        # Standard scaler / numerical cols
        num_features = preprocessor.transformers_[0][2]
        
        # OneHot / categorical cols
        cat_features = []
        try:
            encoder = preprocessor.transformers_[1][1].named_steps['onehot']
            cat_orig_cols = preprocessor.transformers_[1][2]
            cat_features = list(encoder.get_feature_names_out(cat_orig_cols))
        except Exception:
            pass
            
        all_features = list(num_features) + list(cat_features)
        
        importances = None
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
        elif hasattr(classifier, 'coef_'):
            importances = np.abs(classifier.coef_[0])
            
        if importances is not None and len(importances) == len(all_features):
            plt.figure(figsize=(8, 6))
            # Sort importances
            indices = np.argsort(importances)[-15:] # Top 15 features
            plt.barh(range(len(indices)), importances[indices], color='#3b82f6', align='center')
            plt.yticks(range(len(indices)), [all_features[i] for i in indices])
            plt.xlabel('Relative Importance')
            plt.title('Top 15 Feature Importances', fontsize=14, color='#f8fafc', weight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150, facecolor='#0f172a')
            plt.close()
    except Exception as e:
        print(f"Skipping feature importance plot: {e}")

def run_evaluation():
    model_path = 'models/fraud_model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError("Trained model file not found. Run training first.")
        
    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)
        
    pipeline = saved_data['pipeline']
    print(f"Loaded model: {saved_data['model_name']}")
    
    # Load and preprocess test data
    test_df = pd.read_csv('data/processed/test.csv')
    test_df = add_engineered_features(test_df)
    
    X_test = test_df.drop(columns=['fraud_reported'])
    y_test = test_df['fraud_reported']
    
    # Generate predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save plots
    static_images_dir = 'static/images'
    os.makedirs(static_images_dir, exist_ok=True)
    
    # Run EDA on full dataset
    full_df = pd.read_csv('insurance_claims.csv')
    full_df = add_engineered_features(full_df)
    generate_eda_plots(full_df, static_images_dir)
    
    # Run evaluation plots
    generate_evaluation_plots(y_test, y_pred, y_prob, static_images_dir, pipeline, X_test.columns)
    print(f"All plots saved to {static_images_dir}")

if __name__ == '__main__':
    run_evaluation()
