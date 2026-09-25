"""
===========================================================
Final Threat Risk Engine

Input:
final_supervised_drift.csv

Models:
- Adaptive XGBoost
- Isolation Forest
- ERS

Output:
risk assessment

===========================================================
"""


from pathlib import Path

import pandas as pd

import json

import joblib

import xgboost as xgb

import importlib.util


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==========================================================
# LOAD RISK MODULE
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


risk_module = load_module(

    "risk_engine",

    PROJECT_ROOT
    /
    "src"
    /
    "risk"
    /
    "risk_engine.py"

)


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("="*70)

    print(
        "Final Threat Risk Assessment Engine"
    )

    print("="*70)

    # -------------------------------
    # Load Adaptive XGBoost
    # -------------------------------

    model = xgb.XGBClassifier()

    model.load_model(

        PROJECT_ROOT
        /
        "models"
        /
        "adaptive"
        /
        "adaptive_xgboost.json"

    )

    print(
        "Adaptive XGBoost loaded"
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
        "Isolation Forest loaded"
    )

    # -------------------------------
    # Load ERS
    # -------------------------------

    with open(

        PROJECT_ROOT
        /
        "results"
        /
        "ers"
        /
        "ers_report.json"

    ) as f:

        ers_data = json.load(f)

    ers_score = ers_data["ERS"]

    print(

        "ERS:",

        ers_score

    )

    # -------------------------------
    # Load evaluation stream
    # -------------------------------

    data_file = (

        PROJECT_ROOT
        /
        "data"
        /
        "processed"
        /
        "stream"
        /
        "final_supervised_drift.csv"

    )

    df = pd.read_csv(

        data_file,

        nrows=5000

    )

    X = df.drop(

        columns=["target"]

    )

    print(

        "Samples:",

        len(X)

    )

    # -------------------------------
    # Predictions
    # -------------------------------

    probability = (

        model

        .predict_proba(

            X

        )[:, 1]

    )

    anomaly = (

        isolation_model

        .decision_function(

            X

        )

    )

    # -------------------------------
    # Risk Calculation
    # -------------------------------

    results = []

    for i in range(len(X)):

        risk_score = risk_module.calculate_risk_score(

            probability[i],

            anomaly[i],

            ers_score

        )

        level = risk_module.classify_risk(

            risk_score

        )

        results.append(

            {

                "sample_id":

                i,


                "attack_probability":

                float(probability[i]),


                "anomaly_score":

                float(anomaly[i]),


                "ERS":

                float(ers_score),


                "risk_score":

                float(risk_score),


                "threat_level":

                level

            }

        )

    result_df = pd.DataFrame(

        results

    )

    # -------------------------------
    # Save
    # -------------------------------

    output = (

        PROJECT_ROOT
        /
        "results"
        /
        "risk"

    )

    output.mkdir(

        parents=True,

        exist_ok=True

    )

    result_df.to_csv(

        output /
        "final_risk_assessment.csv",

        index=False

    )

    report = {


        "samples":

            len(result_df),


        "risk_distribution":

            result_df["threat_level"]

            .value_counts()

            .to_dict()

    }

    with open(

        output /
        "final_risk_report.json",

        "w"

    ) as f:

        json.dump(

            report,

            f,

            indent=4

        )

    print()

    print(report)

    print()

    print(
        "Final Risk Engine Completed Successfully"
    )


if __name__ == "__main__":

    main()
