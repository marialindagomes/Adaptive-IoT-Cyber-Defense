"""
===========================================================
Adaptive Retraining Pipeline

Input:
final_supervised_drift.csv

Drift:
ADWIN detected point

Output:
adaptive model

===========================================================
"""


from pathlib import Path

import pandas as pd

import json

import importlib.util


PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


adaptive = load_module(

    "adaptive_retrainer",

    PROJECT_ROOT
    /
    "src"
    /
    "models"
    /
    "adaptive_retrainer.py"

)


def main():

    print("="*70)

    print(
        "Adaptive Retraining"
    )

    print("="*70)

    # ------------------------------------------------------
    # Load drift point
    # ------------------------------------------------------

    drift_file = (

        PROJECT_ROOT
        /
        "results"
        /
        "drift"
        /
        "final_supervised_drift_events.csv"

    )

    drift_df = pd.read_csv(

        drift_file

    )

    drift_point = int(

        drift_df.iloc[0]["drift_index"]

    )

    print(

        "Drift point:",

        drift_point

    )

    # ------------------------------------------------------
    # Load retraining window
    # ------------------------------------------------------

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

    window_size = 200000

    start = drift_point

    end = drift_point + window_size

    print(

        "Retraining window:",

        start,

        end

    )

    df = pd.read_csv(

        data_file,

        skiprows=range(

            1,

            start+1

        ),

        nrows=window_size

    )

    print(

        "Training samples:",

        len(df)

    )

    X, y = adaptive.prepare_retraining_data(

        df

    )

    # ------------------------------------------------------
    # Retrain
    # ------------------------------------------------------

    output_dir = (

        PROJECT_ROOT
        /
        "models"
        /
        "adaptive"

    )

    output_dir.mkdir(

        parents=True,

        exist_ok=True

    )

    model_path = (

        output_dir
        /
        "adaptive_xgboost.json"

    )

    adaptive.retrain_xgboost(

        X,

        y,

        output_path=str(model_path)

    )

    report = {


        "drift_point":

            drift_point,


        "retraining_samples":

            len(df),


        "model":

            str(model_path)

    }

    result_dir = (

        PROJECT_ROOT
        /
        "results"
        /
        "adaptive"

    )

    result_dir.mkdir(

        parents=True,

        exist_ok=True

    )

    with open(

        result_dir /
        "retraining_report.json",

        "w"

    ) as f:

        json.dump(

            report,

            f,

            indent=4

        )

    print()

    print(
        "Adaptive Retraining Completed Successfully"
    )


if __name__ == "__main__":

    main()
