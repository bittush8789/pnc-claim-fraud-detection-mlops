# ShieldGuard: System Design & Architecture Document

This document provides a detailed breakdown of the system design, directory structure, and data flow architecture for the P&C Insurance Claim Fraud Detection Platform.

---

## 🏛️ System Architecture

ShieldGuard utilizes a modular machine learning pipeline integrated with a lightweight web interface.

```mermaid
graph TD
    %% Dataset generation
    subgraph Data Layer
        A[generate_data.py] -->|Simulates Claims| B[(insurance_claims.csv)]
    end

    %% Pipeline processing
    subgraph Pipeline Layer (src/)
        B --> C[data_ingestion.py]
        C -->|Raw & Clean Splits| D[feature_engineering.py]
        D -->|Add Risk Ratios| E[preprocessing.py]
        E -->|Standardize & Scale| F[train.py]
        F -->|Selects Best RF/LGBM| G[evaluate.py]
        G -->|Visual Diagnostic Charts| H[(models/fraud_model.pkl)]
    end

    %% Inference and UI serving
    subgraph Service & UI Layer
        H --> I[predict.py]
        I -->|Risk Scoring Engine| J[app.py]
        J -->|Serves Templates| K[index.html / dashboard]
        J -->|Risk Form & Explanations| L[predict.html / result.html]
    end
    
    classDef main fill:#3b82f6,stroke:#1d4ed8,color:#fff;
    classDef ml fill:#8b5cf6,stroke:#6d28d9,color:#fff;
    classDef ui fill:#10b981,stroke:#047857,color:#fff;
    class A,B main;
    class C,D,E,F,G,H,I ml;
    class J,K,L ui;
```

---

## 📂 Folder Structure

```text
pnc-claim-fraud-detection-ml/
├── data/                       # Dataset directories
│   ├── raw/                    # Cached raw CSV dataset
│   └── processed/              # Stratified train.csv and test.csv
├── notebooks/                  # Interactive research
│   └── eda.ipynb               # Jupyter notebook for exploratory data analysis
├── src/                        # Machine Learning Pipeline Modules
│   ├── data_ingestion.py       # Script for data ingestion and validation
│   ├── preprocessing.py        # Pipelines for imputation, OHE, and scaling
│   ├── feature_engineering.py  # Custom fraud indicators and mathematical ratios
│   ├── train.py                # Hyperparameter tuning and model cross-validation
│   ├── evaluate.py             # Model evaluations and plot generation
│   └── predict.py              # Operational inference and risk scoring module
├── models/                     # Model registry
│   └── fraud_model.pkl         # Serialized processor & estimator pipeline
├── templates/                  # Frontend markup templates
│   ├── index.html              # Main KPI dashboard and EDA plots
│   ├── predict.html            # Claim input analysis form
│   └── result.html             # Analysis results with gauge and visual explanation
├── static/                     # Static Web Assets
│   ├── css/
│   │   └── style.css           # Premium glassmorphic UI stylesheet
│   └── images/                 # Matplotlib generated diagrams & charts
├── app.py                      # Flask web application entrypoint
├── generate_data.py            # Synthetic dataset generator
├── requirements.txt            # Python environment dependencies
└── README.md                   # Platform documentation
```

---

## 🔄 Core Components & Data Flow

### 1. Data Ingestion & Validation (`src/data_ingestion.py`)
- **Action**: Loads `insurance_claims.csv`, validates that all 19 required columns exist, removes any duplicates, and divides the dataset into train and test splits (80/20 ratio) using stratified sampling based on the target `fraud_reported` column.
- **Output**: Saves splits to `data/processed/train.csv` and `data/processed/test.csv`.

### 2. Feature Engineering Pipeline (`src/feature_engineering.py`)
Applies deterministic mathematical transformations to surface fraud patterns:
- **`claim_to_insured_ratio`**: Calculates claim value relative to coverage limits. Large values signal maximum damage extraction.
- **`claim_to_premium_ratio`**: Relates claim to annual premiums to spot highly unprofitable accounts.
- **`previous_claim_frequency`**: Calculates claims per year of policy age.
- **`delay_risk_score`**: Scores claim delay metrics (0 delay or > 15 days indicates higher risk).
- **`customer_risk_score`**: Identifies new policies (< 90 days) owned by young drivers (< 25).
- **`incident_risk_score`**: Scores major accidents that lack third-party verification (no police reports or no witnesses).

### 3. Preprocessing ColumnTransformer (`src/preprocessing.py`)
Transforms categorical and numerical features into matrices ready for ML:
- **Numerical Pipeline**: Median Imputation $\rightarrow$ StandardScaler.
- **Categorical Pipeline**: Most Frequent Imputation $\rightarrow$ One-Hot Encoding (handles unknown categories gracefully).

### 4. Training Engine (`src/train.py`)
Trains five models simultaneously:
1. **Logistic Regression** (Linear baseline)
2. **Decision Tree** (Non-linear baseline)
3. **Random Forest** (Ensemble bagging)
4. **XGBoost** (Gradient boosting)
5. **LightGBM** (Histogram-based gradient boosting)

Automatically tunes hyperparameters using 5-fold cross-validation and selects the best estimator using a composite index:
$$\text{Composite Score} = 0.6 \times \text{Recall} + 0.4 \times \text{ROC-AUC}$$

### 5. Prediction & Risk Engine (`src/predict.py`)
Acts as the bridge between model serialization and web requests:
- Unpickles the pipeline and evaluates incoming dictionary data.
- Maps probability into risk levels:
  - $0\% - 40\%$ $\rightarrow$ **LOW RISK** (Standard processing)
  - $41\% - 70\%$ $\rightarrow$ **MEDIUM RISK** (Flagged for review)
  - $71\% - 100\%$ $\rightarrow$ **HIGH RISK** (Immediate SIU Routing)

### 6. Flask Web Server (`app.py`)
Serves as the host layer, reading data, calculating statistics, running predictions on user input, and serving Web pages.
