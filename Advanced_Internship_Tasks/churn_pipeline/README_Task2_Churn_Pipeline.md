# 📉 Customer Churn Prediction — End-to-End ML Pipeline

> **DevelopersHub Corporation — AI/ML Engineering Advanced Internship | Task 2**

A production-ready, reusable **scikit-learn Pipeline** that predicts customer churn for a telecom company. Built on the IBM Telco Customer Churn dataset with full EDA, preprocessing, model training, hyperparameter tuning, and a standalone prediction script — all exportable as a single `.joblib` file.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline Architecture](#pipeline-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Results](#results)
- [predict.py Usage](#predictpy-usage)
- [Skills Gained](#skills-gained)

---

## Overview

This project builds a complete, end-to-end machine learning pipeline for binary classification (churn vs. no churn). The entire workflow — data cleaning, feature encoding, scaling, and model inference — is bundled into a single scikit-learn `Pipeline` object exported with `joblib`. This means new raw customer data can be passed directly to `.predict()` with no separate preprocessing code needed.

**Key highlights:**
- Full EDA with 5 visualizations (class distribution, feature distributions, churn by contract type, churn by internet service, correlation heatmap)
- `ColumnTransformer` handles numeric and categorical features in one step
- Two models compared: Logistic Regression vs. Random Forest
- `GridSearchCV` with 5-fold cross-validation for both models
- Best pipeline exported as a single `.joblib` file
- `predict.py` accepts raw customer data via CLI (default examples, `--json`, or `--csv`)

---

## Dataset

**IBM Telco Customer Churn** — Loaded directly from GitHub raw CSV

| Property | Value |
|---|---|
| Rows | 7,043 customers |
| Features | 20 input columns |
| Target | `Churn` (Yes / No) |
| Class balance | 73.5% No Churn / 26.5% Churn |

**Feature categories:**

| Type | Features |
|---|---|
| Numeric | `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen` |
| Categorical | `gender`, `Partner`, `Contract`, `InternetService`, `PaymentMethod`, + 10 more |

**Key churn drivers found in EDA:**
- Month-to-month contracts churn at ~42% vs ~11% for two-year contracts
- Fiber optic internet customers churn more than DSL customers
- Lower tenure = higher churn probability
- Higher monthly charges = higher churn probability

---

## Pipeline Architecture

```
Raw Customer Data (DataFrame)
        │
        ▼
┌─────────────────────────────────────┐
│         ColumnTransformer           │
│                                     │
│  Numeric cols:                      │
│    SimpleImputer(strategy=median)   │
│    → StandardScaler()               │
│                                     │
│  Categorical cols:                  │
│    SimpleImputer(strategy=mode)     │
│    → OneHotEncoder(handle_unknown   │
│        ="ignore")                   │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Classifier         │
│  (Best of:          │
│   LogisticRegression│
│   RandomForest)     │
└─────────────────────┘
        │
        ▼
  Churn Prediction (0 or 1)
  + Probability Score
```

---

## Project Structure

```
churn_pipeline/
│
├── task2_churn_pipeline.ipynb      # Full EDA + training + evaluation notebook
├── predict.py                      # Standalone prediction script
├── requirements.txt                # All dependencies
│
├── churn_pipeline.joblib           # Exported trained pipeline (ready to use)
└── churn_pipeline_metadata.joblib  # Model metadata and test metrics
```

---

## Installation

**Requirements:** Python 3.9–3.11

```bash
cd churn_pipeline

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** If you get a version mismatch warning when running `predict.py`, just run the notebook once to retrain the pipeline on your local scikit-learn version. It takes about 3–5 minutes.

---

## How to Run

### Option A — Use the pre-trained pipeline (instant, no training needed)

```bash
python predict.py
```

Runs immediately on 3 built-in example customers and prints predictions.

### Option B — Retrain the full pipeline (recommended for your environment)

**On Google Colab:**
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `task2_churn_pipeline.ipynb`
3. Runtime → Run All (~5 minutes)

**Locally:**
```bash
jupyter notebook
# Open task2_churn_pipeline.ipynb → Cell → Run All
```

This will:
- Load and clean the Telco dataset
- Run full EDA with visualizations
- Train Logistic Regression and Random Forest with GridSearchCV
- Compare both models
- Export a fresh `churn_pipeline.joblib` matched to your sklearn version

---

## Results

**Model comparison on held-out test set (1,409 customers):**

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** ✅ | **0.8048** | **0.6552** | **0.5588** | **0.6032** | **0.8412** |
| Random Forest | 0.8020 | 0.6557 | 0.5348 | 0.5891 | 0.8360 |

> Logistic Regression wins slightly because churn drivers in this dataset (contract type, tenure, monthly charges) are largely linear — a realistic and expected result, not a bug.

**Best hyperparameters found by GridSearchCV:**

| Model | Best Params |
|---|---|
| Logistic Regression | `C=10`, `penalty=l2`, `solver=lbfgs` |
| Random Forest | `max_depth=10`, `n_estimators=100`, `min_samples_leaf=1` |

---

## predict.py Usage

The prediction script accepts raw customer data in three ways:

### 1. Built-in demo examples (no arguments needed)
```bash
python predict.py
```

### 2. Single customer via JSON
```bash
python predict.py --json '{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.5,
  "TotalCharges": 1026.0
}'
```

### 3. Batch predictions from a CSV file
```bash
python predict.py --csv new_customers.csv
```

### 4. Save predictions to a CSV file
```bash
python predict.py --csv new_customers.csv --output predictions.csv
```

**Example output:**
```
======================================================================
CHURN PREDICTION RESULTS
======================================================================
   tenure        Contract  MonthlyCharges Churn_Prediction  Churn_Probability
0       1  Month-to-month           29.85              Yes             0.6189
1      65        Two year           95.50               No             0.1071
2       3  Month-to-month           89.10              Yes             0.6366
======================================================================

Summary: 2 of 3 customer(s) predicted to churn (66.7%).
```

---

## Skills Gained

- ✅ End-to-end ML pipeline construction with scikit-learn
- ✅ Exploratory Data Analysis (EDA) with visualizations
- ✅ Feature engineering with `ColumnTransformer`
- ✅ Hyperparameter tuning with `GridSearchCV`
- ✅ Model comparison and evaluation (accuracy, F1, ROC-AUC)
- ✅ Model export and reusability with `joblib`
- ✅ Production-ready prediction script with CLI interface

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green)

| Library | Version | Purpose |
|---|---|---|
| `scikit-learn` | ≥1.3.0 | Pipeline, models, GridSearchCV |
| `pandas` | ≥2.0.0 | Data loading and manipulation |
| `numpy` | ≥1.24.0 | Numerical operations |
| `matplotlib` | ≥3.7.0 | Visualizations |
| `seaborn` | ≥0.12.0 | Statistical plots |
| `joblib` | ≥1.3.0 | Pipeline export and loading |

---

*DevelopersHub Corporation — AI/ML Engineering Internship*
