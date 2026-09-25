"""
===========================================================
Streaming Feature Alignment (Memory Optimized)

Purpose:
Align large stream dataset without loading
whole file into RAM.

===========================================================
"""


from pathlib import Path

import pandas as pd

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


alignment_module = load_module(

    "feature_alignment",

    PROJECT_ROOT
    /
    "src"
    /
    "data"
    /
    "feature_alignment.py"

)


load_feature_list = (

    alignment_module

    .load_feature_list

)


align_features = (

    alignment_module

    .align_features

)


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("="*70)

    print(
        "Memory Optimized Stream Feature Alignment"
    )

    print("="*70)

    input_file = (

        PROJECT_ROOT
        /
        "data"
        /
        "processed"
        /
        "stream"
        /
        "stream_data.csv"

    )

    feature_file = (

        PROJECT_ROOT
        /
        "results"
        /
        "feature_processing"
        /
        "ciciot_features.json"

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
        "aligned_stream.csv"

    )

    required_features = load_feature_list(

        feature_file

    )

    print(

        "Required features:",

        len(required_features)

    )

    chunk_size = 10000

    first_write = True

    total_rows = 0

    print(

        "Processing chunks..."

    )

    for chunk in pd.read_csv(

            input_file,

            chunksize=chunk_size,

            low_memory=False

    ):

        aligned_chunk = align_features(

            chunk,

            required_features

        )

        aligned_chunk.to_csv(

            output_file,

            mode="w" if first_write else "a",

            header=first_write,

            index=False

        )

        first_write = False

        total_rows += len(

            aligned_chunk

        )

        print(

            "Processed rows:",

            total_rows

        )

        del chunk

        del aligned_chunk

    print("\n")

    print("="*70)

    print(
        "Feature Alignment Completed Successfully"
    )

    print("="*70)

    print(

        "Saved:",

        output_file

    )

    print(

        "Total rows:",

        total_rows

    )


if __name__ == "__main__":

    main()
