"""
===========================================================
Save Processed CICIoT2023 Dataset

Purpose:
- Recreate preprocessing pipeline
- Save processed train/validation/test datasets

Output:
data/processed/ciciot2023/

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


# ==========================================================
# IMPORT FUNCTIONS
# ==========================================================

label_module = load_module(

    "label_mapper",

    PROJECT_ROOT /
    "src/data/label_mapper.py"

)


clean_module = load_module(

    "clean_data",

    PROJECT_ROOT /
    "src/data/clean_data.py"

)


feature_module = load_module(

    "feature_processing",

    PROJECT_ROOT /
    "src/data/feature_processing.py"

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
# LOAD FILES
# ==========================================================

def load_split_files(split_name):

    with open(

        PROJECT_ROOT /
        "results/split"
        /
        f"{split_name}_files.json"

    ) as f:

        return json.load(f)


def load_csv_files(files, sample_rows=50000):

    data_path = (

        PROJECT_ROOT
        /
        "data/raw/ciciot2023/csv"

    )

    dfs = []

    for file in files:

        print(
            "Reading:",
            file
        )

        df = pd.read_csv(

            data_path / file,

            nrows=sample_rows

        )

        dfs.append(df)

    return pd.concat(

        dfs,

        ignore_index=True

    )


# ==========================================================
# PROCESS FUNCTION
# ==========================================================

def process_dataset(df, name):

    print("\n")

    print("="*60)

    print(
        f"Processing {name}"
    )

    print("="*60)

    # Label mapping

    df = map_labels(df)

    # Cleaning

    df = clean_dataset(

        df,

        remove_cols=[

            "label"

        ]

    )

    # Feature processing

    df = process_features(

        df

    )

    print(

        "Final shape:",

        df.shape

    )

    return df


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("="*70)

    print(
        "Saving Processed CICIoT2023 Dataset"
    )

    print("="*70)

    train_files = load_split_files(

        "train"

    )

    val_files = load_split_files(

        "validation"

    )

    test_files = load_split_files(

        "test"

    )

    train_df = load_csv_files(

        train_files

    )

    val_df = load_csv_files(

        val_files

    )

    test_df = load_csv_files(

        test_files

    )

    train_processed = process_dataset(

        train_df,

        "TRAIN"

    )

    val_processed = process_dataset(

        val_df,

        "VALIDATION"

    )

    test_processed = process_dataset(

        test_df,

        "TEST"

    )

    # Save path

    output = (

        PROJECT_ROOT
        /
        "data"
        /
        "processed"
        /
        "ciciot2023"

    )

    output.mkdir(

        parents=True,

        exist_ok=True

    )

    train_processed.to_csv(

        output /
        "train_features.csv",

        index=False

    )

    val_processed.to_csv(

        output /
        "validation_features.csv",

        index=False

    )

    test_processed.to_csv(

        output /
        "test_features.csv",

        index=False

    )

    print("\nSaved files:")

    print(

        output /
        "train_features.csv"

    )

    print(

        output /
        "validation_features.csv"

    )

    print(

        output /
        "test_features.csv"

    )

    print("\nProcessed Dataset Saved Successfully")


if __name__ == "__main__":

    main()
