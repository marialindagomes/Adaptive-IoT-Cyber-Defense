"""
===========================================================
Final Supervised ADWIN Drift Detection

Input:
final_supervised_drift.csv

Output:
results/drift/

===========================================================
"""


from pathlib import Path

import pandas as pd

import json

import joblib

import importlib.util


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==========================================================
# LOAD ADWIN MODULE
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


adwin_module = load_module(

    "adwin_detector",

    PROJECT_ROOT
    /
    "src"
    /
    "drift"
    /
    "adwin_detector.py"

)


detect_drift = (

    adwin_module

    .detect_drift

)


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("="*70)

    print(
        "Final Supervised ADWIN Drift Detection"
    )

    print("="*70)

    # ------------------------------------------------------
    # Load model
    # ------------------------------------------------------

    model_path = (

        PROJECT_ROOT
        /
        "models"
        /
        "xgboost"
        /
        "calibrated_xgboost.pkl"

    )

    model = joblib.load(

        model_path

    )

    print(
        "Calibrated XGBoost loaded"
    )

    # ------------------------------------------------------
    # Stream file
    # ------------------------------------------------------

    stream_file = (

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

    chunk_size = 10000

    error_stream = []

    total_samples = 0

    print(
        "Processing stream..."
    )

    for chunk in pd.read_csv(

        stream_file,

        chunksize=chunk_size,

        low_memory=False

    ):

        y_true = chunk["target"]

        X = chunk.drop(

            columns=["target"]

        )

        probability = (

            model

            .predict_proba(

                X

            )[:, 1]

        )

        prediction = (

            probability >= 0.5

        ).astype(int)

        error = (

            prediction != y_true.values

        ).astype(int)

        error_stream.extend(

            error

        )

        total_samples += len(chunk)

        print(

            "Processed:",

            total_samples

        )

    print()

    print(

        "Total samples:",

        total_samples

    )

    print(

        "Total errors:",

        sum(error_stream)

    )

    # ------------------------------------------------------
    # ADWIN
    # ------------------------------------------------------

    print()

    print(
        "Running ADWIN..."
    )

    drift_points = detect_drift(

        error_stream

    )

    print(

        "Detected drift points:",

        len(drift_points)

    )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    output_dir = (

        PROJECT_ROOT
        /
        "results"
        /
        "drift"

    )

    output_dir.mkdir(

        parents=True,

        exist_ok=True

    )

    pd.DataFrame(

        {

            "drift_index":

                drift_points

        }

    ).to_csv(

        output_dir
        /
        "final_supervised_drift_events.csv",

        index=False

    )

    report = {


        "experiment":

            "final_supervised_unknown_drift",


        "total_samples":

            total_samples,


        "total_errors":

            int(sum(error_stream)),


        "drift_count":

            len(drift_points),


        "drift_points":

            drift_points

    }

    with open(

        output_dir
        /
        "final_supervised_drift_report.json",

        "w"

    ) as f:

        json.dump(

            report,

            f,

            indent=4

        )

    print()

    print(

        "Saved:",

        output_dir

    )

    print(

        "Final Supervised ADWIN Completed Successfully"

    )


if __name__ == "__main__":

    main()
