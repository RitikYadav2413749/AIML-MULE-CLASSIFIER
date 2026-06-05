# AIML Mule Account Classifier

> **AI/ML-Based Fraud Detection Framework for Mule Account Classification**

---

## Project Overview

This project builds a comprehensive AI/ML classification system that identifies
suspicious and mule accounts using transaction data. Mule accounts are used by
fraudsters to receive, transfer, and hide illegally obtained funds through
multiple banking channels.

The system analyzes customer transaction behavior and account activity patterns
to distinguish fraudulent accounts from legitimate ones using **10 advanced
strategies** spanning behavioral analysis, graph intelligence, deep learning
anomaly detection, and explainable AI.

### Target Variable

- **Feature 3924 (F3924)** — Binary classification: Suspicious (1) vs Legitimate (0)

### Hint Features (18 commonly used fraud detection features)

`F115`, `F321`, `F527`, `F531`, `F670`, `F1692`, `F2082`, `F2122`, `F2582`,
`F2678`, `F2737`, `F2956`, `F3043`, `F3836`, `F3887`, `F3889`, `F3891`, `F3894`

---

## 10 Core Strategies

| #  | Strategy                                            | Module(s)                                      |
|----|-----------------------------------------------------|------------------------------------------------|
| 1  | Behavioral DNA Fingerprinting                       | `src/feature_engineering/behavioral_dna.py`    |
| 2  | Peer Group Anomaly Detection                        | `src/feature_engineering/peer_group_anomaly.py`|
| 3  | Guilt by Association (Graph Propagation + GNN)      | `src/graph/*`                                  |
| 4  | Autoencoder Anomaly Detection                       | `src/feature_engineering/autoencoder_features.py`, `src/models/autoencoder_model.py` |
| 5  | Temporal Drift Detection (PSI Score)                | `src/feature_engineering/temporal_drift.py`    |
| 6  | CTGAN Synthetic Data Augmentation                   | `src/models/ctgan_augmentation.py`             |
| 7  | Meta-Feature Stacking with Diversity (7 models)     | `src/models/meta_stacking.py`                  |
| 8  | Meta-XAI Features (SHAP as features)                | `src/feature_engineering/meta_xai_features.py` |
| 9  | Business Cost Threshold Optimization                | `src/models/cost_optimization.py`              |
| 10 | Live Real-Time Alert Dashboard (Streamlit)          | `dashboard/*`                                  |

---

## Project Structure

```
AIML Mule classifier/
├── config/                    → Central configuration (paths, constants, hyperparameters)
├── data/                      → Raw, processed, enriched, and synthetic data
├── notebooks/                 → Day-wise execution scripts (Days 1–15)
├── src/                       → Core source code modules
│   ├── eda/                   → Exploratory Data Analysis (7 modules)
│   ├── preprocessing/         → Data cleaning, splitting, scaling (5 modules)
│   ├── feature_engineering/   → Feature engineering strategies (7 modules)
│   ├── graph/                 → Graph intelligence — NetworkX, PyVis, GNN (6 modules)
│   ├── models/                → Model training, tuning, evaluation (7 modules)
│   ├── explainability/        → SHAP analysis, alerts, report generation (4 modules)
│   └── utils/                 → Shared utilities — I/O, metrics, logging (3 modules)
├── dashboard/                 → Streamlit multi-page dashboard application
│   ├── pages/                 → 5 dashboard pages
│   ├── components/            → 4 reusable UI components
│   └── assets/                → Custom CSS styling
├── presentation/              → 15-slide deck outline & generator
├── reports/                   → Generated EDA reports, metrics JSON, figures
├── saved_models/              → Trained model artifacts (pkl, h5, pt, json)
├── logs/                      → Execution and training logs
└── run_pipeline.py            → Master orchestration and execution runner
```

See [`PROJECT_STRUCTURE.txt`](PROJECT_STRUCTURE.txt) for the full annotated directory tree.

---

## Tech Stack

| Category           | Libraries                                                  |
|--------------------|------------------------------------------------------------|
| Language           | Python 3.10+                                               |
| Core ML            | XGBoost, LightGBM, CatBoost, scikit-learn                  |
| Deep Learning      | TensorFlow / Keras (Autoencoder, VAE)                      |
| Imbalance          | imbalanced-learn (SMOTE)                                   |
| Graph              | NetworkX, PyVis, python-louvain; PyTorch Geometric (optional)|
| Data Augmentation  | CTGAN                                                      |
| Explainability     | SHAP                                                       |
| Tuning             | Optuna                                                     |
| Dashboard          | Streamlit                                                  |
| Visualization      | Plotly, Seaborn, Matplotlib                                |
| Reporting          | FPDF, openpyxl                                             |
| EDA                | ydata-profiling, pandas                                    |
| Presentation       | python-pptx (optional)                                     |

---

## Expected Performance Targets

| Metric            | Baseline | Target   |
|-------------------|----------|----------|
| AUC-ROC           | 0.75     | > 0.92   |
| AUC-PR            | 0.40     | > 0.75   |
| F1 Score          | 0.55     | > 0.78   |
| Recall (Fraud)    | 0.60     | > 0.85   |
| Precision (Fraud) | 0.50     | > 0.72   |
| Business Cost     | High     | ~60% reduction |

---

## Setup Instructions

```bash
# 1. Clone repository
git clone <repo-url>
cd "AIML Mule classifier"

# 2. Create virtual environment
python -m venv venv

# 3. Activate (Windows)
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the master pipeline (automatic mock generation if no custom raw/transactions.csv placed)
python run_pipeline.py

# 6. Launch the Streamlit alert investigation dashboard
streamlit run dashboard/app.py
```

---

## Key Design Decisions

- **No data leakage**: SMOTE and scaling are applied *only* on training data.
  Validation/test sets are never touched by SMOTE.
- **Optional heavy dependencies**: GNN (PyTorch Geometric), Autoencoder (TensorFlow),
  CTGAN, and Optuna are wrapped in try/except; the system degrades gracefully.
- **Central configuration**: All paths, constants, and hyperparameters live in
  `config/config.py` — no hardcoded values across modules.
- **Reproducibility**: All random operations use `RANDOM_STATE = 42`.
