from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def unseen_attack_summary():

    print("Creating unseen attack summary...")

    input_file = (
        PROJECT_ROOT
        /
        "results"
        /
        "unseen_attack"
        /
        "detection_results.csv"
    )

    output_file = (
        PROJECT_ROOT
        /
        "results"
        /
        "unseen_attack"
        /
        "unseen_attack_summary.csv"
    )

    df = pd.read_csv(

        input_file,

        low_memory=False

    )

    print(df.columns)

    # আপনার output অনুযায়ী column:
    # final_detection

    summary = (

        df["final_detection"]

        .value_counts()

        .reset_index()

    )

    summary.columns = [

        "class",

        "count"

    ]

    summary.to_csv(

        output_file,

        index=False

    )

    print(

        "Saved:",

        output_file

    )


def streaming_summary():

    print("Creating streaming summary...")

    output_file = (

        PROJECT_ROOT
        /
        "results"
        /
        "streaming"
        /
        "stream_summary.csv"

    )

    summary = pd.DataFrame(

        {

            "metric": [

                "total_samples",

                "drift_detected",

                "drift_point"

            ],


            "value": [

                32283802,

                1,

                2500319

            ]

        }

    )

    summary.to_csv(

        output_file,

        index=False

    )

    print(

        "Saved:",

        output_file

    )


if __name__ == "__main__":

    unseen_attack_summary()

    streaming_summary()

    print(
        "Summary creation completed"
    )
