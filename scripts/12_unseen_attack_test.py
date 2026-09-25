"""
===========================================================
Unseen Attack Test

Uses:
- Calibrated XGBoost
- Isolation Forest

===========================================================
"""


from pathlib import Path
import importlib.util

import pandas as pd

import joblib


# ==========================================================
# PATH
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==========================================================
# LOAD MODULE
# ==========================================================

def load_module(name, path):

    spec = importlib.util.spec_from_file_location(

        name,

        path

    )

    module = importlib.util.module_from_spec(

        spec

    )

    spec.loader.exec_module(

        module

    )

    return module


detector = load_module(

    "unknown_detector",

    PROJECT_ROOT
    /
    "src"
    /
    "models"
    /
    "unknown_attack_detector.py"

)


create_detection_report = (

    detector
    .create_detection_report

)


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("="*60)

    print(
        "Improved Unseen Attack Detection Test"
    )

    print("="*60)

    # -------------------------------
    # Load calibrated XGBoost
    # -------------------------------

    calibrated_model = joblib.load(

        PROJECT_ROOT
        /
        "models"
        /
        "xgboost"
        /
        "calibrated_xgboost.pkl"

    )

    # -------------------------------
    # Load Isolation Forest
    # -------------------------------

    isolation_model = joblib.load(

        PROJECT_ROOT
        /
        "models"
        /
        "anomaly"
        /
        "isolation_forest.pkl"

    )

    print(
        "Models loaded successfully"
    )

    # -------------------------------
    # Load processed test data
    # -------------------------------

    # আমরা আগে তৈরি করা test feature dataset ব্যবহার করব

    test_file = (

        PROJECT_ROOT
        /
        "data"
        /
        "processed"
        /
        "test_features.csv"

    )

    if not test_file.exists():

        raise FileNotFoundError(

            "Processed test dataset not found"

        )

    df = pd.read_csv(

        test_file

    )

    y_true = df["target"]

    X_test = df.drop(

        columns=["target"]

    )

    print(

        "Test samples:",

        len(X_test)

    )

    # -------------------------------
    # XGBoost probability
    # -------------------------------

    xgb_probability = (

        calibrated_model

        .predict_proba(

            X_test

        )[:, 1]

    )

    # -------------------------------
    # Isolation Forest
    # -------------------------------

    anomaly_prediction = (

        isolation_model

        .predict(

            X_test

        )

    )

    anomaly_score = (

        isolation_model

        .decision_function(

            X_test

        )

    )

    # -------------------------------
    # Combine
    # -------------------------------

    result = create_detection_report(

        y_true,

        xgb_probability,

        anomaly_prediction,

        anomaly_score

    )

    output = (

        PROJECT_ROOT
        /
        "results"
        /
        "unseen_attack"

    )

    output.mkdir(

        parents=True,

        exist_ok=True

    )

    result.to_csv(

        output
        /
        "detection_results.csv",

        index=False

    )

    print(

        "Saved:",

        output
        /
        "detection_results.csv"

    )

    print(

        result["final_detection"]

        .value_counts()

    )

    print(

        "Unseen Attack Test Completed Successfully"

    )


if __name__ == "__main__":

    main()
