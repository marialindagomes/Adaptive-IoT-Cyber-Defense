"""
===========================================================
Final Supervised Unknown Drift Generator

Input:
data/processed/ciciot2023/train_features.csv

Output:
data/processed/stream/final_supervised_drift.csv

Purpose:
Create labeled concept drift for ADWIN evaluation.

===========================================================
"""


from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():

    print("="*70)

    print(
        "Creating Final Supervised Drift Stream"
    )

    print("="*70)

    input_file = (

        PROJECT_ROOT
        /
        "data"
        /
        "processed"
        /
        "ciciot2023"
        /
        "train_features.csv"

    )

    output_file = (

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

    # drift after half of training stream

    drift_start = 2500000

    total = 0

    first_write = True

    for chunk in pd.read_csv(

        input_file,

        chunksize=chunk_size,

        low_memory=False

    ):

        start = total

        end = total + len(chunk)

        if end > drift_start:

            print(

                "Injecting unknown drift:",

                start,

                end

            )

            feature_columns = [

                c

                for c in chunk.columns

                if c != "target"

            ]

            # create unseen behaviour

            chunk[feature_columns] = (

                chunk[feature_columns]

                *

                8

            )

            chunk[feature_columns] = (

                chunk[feature_columns]

                +

                chunk[feature_columns]

                .abs()

                .pow(0.25)

            )

        chunk.to_csv(

            output_file,

            mode="w" if first_write else "a",

            header=first_write,

            index=False

        )

        first_write = False

        total = end

        print(

            "Processed:",

            total

        )

    print("\n")

    print("="*70)

    print(
        "Final Supervised Drift Created Successfully"
    )

    print("="*70)

    print(

        "Saved:",

        output_file

    )

    print(

        "Total rows:",

        total

    )


if __name__ == "__main__":

    main()
