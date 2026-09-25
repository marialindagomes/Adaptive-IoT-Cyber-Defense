"""
===========================================================
Streaming Simulation Pipeline

Purpose:
Simulate real-time cyber traffic stream

Models:
- Calibrated XGBoost
- Isolation Forest

Input:
aligned_stream.csv

Output:
stream_predictions.csv

===========================================================
"""


from pathlib import Path

import pandas as pd

import joblib

import xgboost as xgb


# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("="*70)

    print(
        "Real-Time Streaming Simulation"
    )

    print("="*70)

    # ------------------------------------------------------
    # Load models
    # ------------------------------------------------------

    print(
        "Loading models..."
    )

    xgb_model = joblib.load(

        PROJECT_ROOT
        /
        "models"
        /
        "xgboost"
        /
        "calibrated_xgboost.pkl"

    )

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
        "aligned_stream.csv"

    )

    output_dir = (

        PROJECT_ROOT
        /
        "results"
        /
        "streaming"

    )

    output_dir.mkdir(

        parents=True,

        exist_ok=True

    )

    output_file = (

        output_dir
        /
        "stream_predictions.csv"

    )

    # ------------------------------------------------------
    # Streaming
    # ------------------------------------------------------

    chunk_size = 10000

    first_write = True

    total_processed = 0

    print(
        "Starting stream..."
    )

    for chunk in pd.read_csv(

        stream_file,

        chunksize=chunk_size,

        low_memory=False

    ):

        X = chunk.copy()

        # XGBoost probability

        probability = (

            xgb_model

            .predict_proba(

                X

            )[:, 1]

        )

        xgb_prediction = (

            probability >= 0.5

        ).astype(int)

        # Isolation Forest

        anomaly_prediction = (

            isolation_model

            .predict(

                X

            )

        )

        anomaly_score = (

            isolation_model

            .decision_function(

                X

            )

        )

        # Final decision

        final_status = []

        for prob, anomaly in zip(

            probability,

            anomaly_prediction

        ):

            if (

                anomaly == -1

                and

                prob < 0.8

            ):

                status = "Unknown Attack"

            elif prob >= 0.8:

                status = "Known Attack"

            else:

                status = "Normal"

            final_status.append(

                status

            )

        result = pd.DataFrame(

            {

                "sample_id":

                    range(

                        total_processed,

                        total_processed + len(chunk)

                    ),


                "attack_probability":

                    probability,


                "xgb_prediction":

                    xgb_prediction,


                "anomaly_prediction":

                    anomaly_prediction,


                "anomaly_score":

                    anomaly_score,


                "final_status":

                    final_status

            }

        )

        result.to_csv(

            output_file,

            mode="w" if first_write else "a",

            header=first_write,

            index=False

        )

        first_write = False

        total_processed += len(chunk)

        print(

            "Processed samples:",

            total_processed

        )

    print("\n")

    print("="*70)

    print(
        "Streaming Simulation Completed Successfully"
    )

    print("="*70)

    print(

        "Saved:",

        output_file

    )


if __name__ == "__main__":

    main()
