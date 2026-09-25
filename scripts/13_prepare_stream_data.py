"""
===========================================================
Prepare Flow Stream Dataset

Input:
CICIoT2023 flow CSV files

Output:
data/processed/stream/

===========================================================
"""


from pathlib import Path

import pandas as pd

import json


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
        "Preparing Flow Stream Dataset"
    )

    print("="*70)

    input_folder = (

        PROJECT_ROOT
        /
        "data"
        /
        "raw"
        /
        "ciciot2023"
        /
        "pcap"

    )

    files = list(

        input_folder.glob("*.csv")

    )

    if not files:

        raise FileNotFoundError(

            "No flow CSV files found"

        )

    print(

        "Total flow files:",

        len(files)

    )

    dataframes = []

    for file in files:

        print(

            "Reading:",

            file.name

        )

        df = pd.read_csv(

            file,

            low_memory=False

        )

        dataframes.append(

            df

        )

    stream_df = pd.concat(

        dataframes,

        ignore_index=True

    )

    print(

        "\nCombined Shape:",

        stream_df.shape

    )

    # Save location

    output = (

        PROJECT_ROOT
        /
        "data"
        /
        "processed"
        /
        "stream"

    )

    output.mkdir(

        parents=True,

        exist_ok=True

    )

    stream_file = (

        output
        /
        "stream_data.csv"

    )

    stream_df.to_csv(

        stream_file,

        index=False

    )

    metadata = {


        "total_samples":

            len(stream_df),


        "total_features":

            len(stream_df.columns),


        "source_files":

            [

                f.name

                for f in files

            ]

    }

    with open(

        output /
        "stream_metadata.json",

        "w"

    ) as f:

        json.dump(

            metadata,

            f,

            indent=4

        )

    print(

        "\nSaved:",

        stream_file

    )

    print(

        "Stream preparation completed successfully"

    )


if __name__ == "__main__":

    main()
