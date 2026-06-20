"""
predict.py — Production inference script for the Telco Customer Churn pipeline.

Loads the fitted scikit-learn Pipeline exported by task2_churn_pipeline.ipynb
(churn_pipeline.joblib) and runs predictions on new, raw customer records.
Because the exported object is a full Pipeline (ColumnTransformer + classifier),
NO separate preprocessing code is needed here — raw fields go straight in.

Usage examples:
    # 1. Predict on the built-in example customers
    python predict.py

    # 2. Predict on a CSV file of new customers (must contain the required columns)
    python predict.py --csv new_customers.csv

    # 3. Predict on a single customer via inline JSON
    python predict.py --json '{"gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", ...}'
"""

import argparse
import json
import os
import sys

import pandas as pd

try:
    import joblib
except ImportError:
    print("ERROR: joblib is not installed. Run: pip install -r requirements.txt")
    sys.exit(1)


PIPELINE_PATH = "churn_pipeline.joblib"
METADATA_PATH = "churn_pipeline_metadata.joblib"

# The exact raw feature columns the pipeline expects, in the order produced
# by the training notebook (order does not actually matter to scikit-learn's
# ColumnTransformer since it selects columns by name, but we validate presence).
REQUIRED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

# A few example customers for a quick smoke-test / demo run with no arguments.
EXAMPLE_CUSTOMERS = [
    {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85,
    },
    {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 65, "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "Fiber optic", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 95.5, "TotalCharges": 6200.0,
    },
    {
        "gender": "Female", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
        "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 89.1, "TotalCharges": 267.3,
    },
]


def load_pipeline_and_metadata():
    """Load the fitted pipeline (and optional metadata) from disk, with clear errors."""
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(
            f"Could not find '{PIPELINE_PATH}'. Run task2_churn_pipeline.ipynb first "
            f"to train and export the pipeline, or place the .joblib file in this "
            f"directory."
        )

    pipeline = joblib.load(PIPELINE_PATH)

    metadata = None
    if os.path.exists(METADATA_PATH):
        metadata = joblib.load(METADATA_PATH)

    return pipeline, metadata


def validate_dataframe(df: pd.DataFrame) -> None:
    """Raise a clear, actionable error if required columns are missing."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input data is missing required column(s): {missing}\n"
            f"Required columns are: {REQUIRED_COLUMNS}"
        )

    # TotalCharges/MonthlyCharges/tenure/SeniorCitizen must be numeric;
    # coerce gracefully and warn rather than crash on dirty input.
    for col in ["MonthlyCharges", "TotalCharges", "tenure", "SeniorCitizen"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isna().any():
                print(f"WARNING: Non-numeric values found in '{col}' were coerced to NaN "
                      f"and will be median-imputed by the pipeline.")


def predict_churn(df: pd.DataFrame, pipeline) -> pd.DataFrame:
    """Run prediction + probability scoring on a dataframe of raw customer records."""
    predictions = pipeline.predict(df)
    probabilities = pipeline.predict_proba(df)[:, 1]  # probability of class "1" = churn

    results = df.copy()
    results["Churn_Prediction"] = ["Yes" if p == 1 else "No" for p in predictions]
    results["Churn_Probability"] = probabilities.round(4)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Predict customer churn using the trained pipeline (churn_pipeline.joblib)."
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to a CSV file of new customer records (must contain the required columns)."
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="A single customer record as a JSON string, e.g. '{\"gender\": \"Female\", ...}'"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional path to save predictions as a CSV file."
    )
    args = parser.parse_args()

    # --- Load model ---
    try:
        pipeline, metadata = load_pipeline_and_metadata()
        print(f"Loaded pipeline from '{PIPELINE_PATH}'")
        if metadata:
            print(f"Model type: {metadata.get('best_model_name', 'Unknown')}")
            test_metrics = metadata.get("test_metrics", {}).get(metadata.get("best_model_name"), {})
            if test_metrics:
                print(f"Reported test-set performance: "
                      f"Accuracy={test_metrics.get('Accuracy', 0):.4f}, "
                      f"F1={test_metrics.get('F1-Score', 0):.4f}, "
                      f"ROC-AUC={test_metrics.get('ROC-AUC', 0):.4f}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load the pipeline — {e}")
        sys.exit(1)

    # --- Build input dataframe from CLI args, or fall back to built-in examples ---
    try:
        if args.csv:
            if not os.path.exists(args.csv):
                raise FileNotFoundError(f"CSV file not found: {args.csv}")
            input_df = pd.read_csv(args.csv)
            print(f"\nLoaded {len(input_df)} customer record(s) from '{args.csv}'")
        elif args.json:
            try:
                record = json.loads(args.json)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON provided to --json: {e}")
            input_df = pd.DataFrame([record])
            print("\nLoaded 1 customer record from --json input")
        else:
            input_df = pd.DataFrame(EXAMPLE_CUSTOMERS)
            print(f"\nNo --csv or --json provided. Running on {len(input_df)} built-in example "
                  f"customers as a demo.")

        validate_dataframe(input_df)

    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # --- Predict ---
    try:
        results = predict_churn(input_df, pipeline)
    except Exception as e:
        print(f"ERROR: Prediction failed — {e}")
        sys.exit(1)

    # --- Display results ---
    print("\n" + "=" * 70)
    print("CHURN PREDICTION RESULTS")
    print("=" * 70)
    display_cols = ["tenure", "Contract", "MonthlyCharges", "Churn_Prediction", "Churn_Probability"]
    print(results[display_cols].to_string(index=True))
    print("=" * 70)

    n_churn = (results["Churn_Prediction"] == "Yes").sum()
    print(f"\nSummary: {n_churn} of {len(results)} customer(s) predicted to churn "
          f"({n_churn / len(results) * 100:.1f}%).")

    # --- Optionally save to CSV ---
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nFull results saved to: {args.output}")


if __name__ == "__main__":
    main()
