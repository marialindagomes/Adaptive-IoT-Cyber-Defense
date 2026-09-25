"""
===========================================================
ADWIN Drift Detection Pipeline

Input:
stream_predictions.csv

Output:
drift_events.csv

===========================================================
"""


from pathlib import Path

import pandas as pd

import json

import importlib.util


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
        "ADWIN Drift Detection"
    )

    print("="*70)

    prediction_file = (

        PROJECT_ROOT
        /
        "results"
        /
        "streaming"
        /
        "stream_predictions.csv"

    )

    df = pd.read_csv(

        prediction_file

    )

    print(

        "Total predictions:",

        len(df)

    )

    # Error definition

    # এখানে anomaly:
    # XGB prediction confidence change কে error হিসেবে ধরা হচ্ছে

    error_stream = (

        1 -

        df["attack_probability"]

    ).values

    drift_points = detect_drift(

        error_stream

    )

    print(

        "Drift points detected:",

        len(drift_points)

    )

    output = (

        PROJECT_ROOT
        /
        "results"
        /
        "drift"

    )

    output.mkdir(

        parents=True,

        exist_ok=True

    )

    drift_df = pd.DataFrame(

        {

            "drift_index":

                drift_points

        }

    )

    drift_df.to_csv(

        output /
        "drift_events.csv",

        index=False

    )

    report = {


        "total_samples":

            len(df),


        "drift_count":

            len(drift_points),


        "drift_points":

            drift_points

    }

    with open(

        output /
        "drift_report.json",

        "w"

    ) as f:

        json.dump(

            report,

            f,

            indent=4

        )

    print(

        "Saved:",

        output

    )

    print(
        "ADWIN Drift Detection Completed Successfully"
    )


if __name__ == "__main__":

    main()
