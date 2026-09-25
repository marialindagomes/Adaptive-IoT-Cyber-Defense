"""
===========================================================
Prepare Test Features Pipeline

Purpose:
- Load CICIoT2023 test split
- Apply label mapping
- Cleaning
- Feature processing
- Save processed test dataset

Output:
data/processed/test_features.csv

===========================================================
"""


from pathlib import Path

import json

import importlib.util

import pandas as pd


# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==========================================================
# MODULE LOADER
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


# ==========================================================
# LOAD MODULES
# ==========================================================
label_module = load_module(

    "label_mapper",

    PROJECT_ROOT
    /
    "src"
    /
    "data"
    /
    "label_mapper.py"

)


clean_module = load_module(

    "clean_data",

    PROJECT_ROOT
    /
    "src"
    /
    "data"
    /
    "clean_data.py"

)


feature_module = load_module(

    "feature_processing",

    PROJECT_ROOT
    /
    "src"
    /
    "data"
    /
    "feature_processing.py"

)


map_labels = (

    label_module

    .map_ciciot2023_labels

)


clean_dataset = (

    clean_module

    .clean_dataset

)


process_features = (

    feature_module

    .process_features

)


# ==========================================================
# LOAD TEST FILES
# ==========================================================

def load_test_files(sample_rows=50000):

    split_file = (

        PROJECT_ROOT
        /
        "results"
        /
        "split"
        /
        "test_files.json"

    )

    with open(split_file) as f:

        files = json.load(f)

    data_path = (

        PROJECT_ROOT
        /
        "data"
        /
        "raw"
        /
        "ciciot2023"
        /
        "csv"

    )

    dataframes = []

    print(
        "Loading test files..."
    )

    for file in files:

        print(
            "Reading:",
            file
        )

        df = pd.read_csv(

            data_path / file,

            nrows=sample_rows

        )

        dataframes.append(df)

    return pd.concat(

        dataframes,

        ignore_index=True

    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("="*70)

    print(
        "Preparing Test Feature Dataset"
    )

    print("="*70)

    # Load data

    df = load_test_files()

    print(

        "\nOriginal Shape:",

        df.shape

    )

    # Label mapping

    df = map_labels(

        df

    )

    print(

        "\nAfter Label Mapping:"

    )

    print(

        df["target"]

        .value_counts()

    )

    # Cleaning

    df = clean_dataset(

        df,

        remove_cols=[

            "label"

        ]

    )

    print(

        "\nAfter Cleaning:",

        df.shape

    )

    # Feature processing

    df = process_features(

        df

    )

    print(

        "\nFinal Feature Dataset:",

        df.shape

    )

    # Save

    output = (

        PROJECT_ROOT
        /
        "data"
        /
        "processed"

    )

    output.mkdir(

        parents=True,

        exist_ok=True

    )

    df.to_csv(

        output
        /
        "test_features.csv",

        index=False

    )

    print(

        "\nSaved:",

        output
        /
        "test_features.csv"

    )

    print(

        "Test Feature Preparation Completed Successfully"

    )


if __name__ == "__main__":

    main()
