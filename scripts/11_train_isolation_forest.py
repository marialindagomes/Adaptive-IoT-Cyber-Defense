"""
===========================================================
Isolation Forest Training Pipeline

Purpose:
- Train anomaly detector using benign traffic
- Detect abnormal attack behavior

Dataset:
CICIoT2023

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


if_module = load_module(

    "isolation_forest",

    PROJECT_ROOT /
    "src/models/isolation_forest.py"

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


train_isolation_forest = (

    if_module
    .train_isolation_forest

)


predict_anomaly = (

    if_module
    .predict_anomaly

)


# ==========================================================
# LOAD DATA
# ==========================================================

def load_files(

        file_names,

        sample_rows=50000

):

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

    dfs = []

    for file in file_names:

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
# MAIN
# ==========================================================

def main():

    print("="*70)

    print(
        "Isolation Forest Unknown Attack Detection"
    )

    print("="*70)

    # Load train files

    with open(

        PROJECT_ROOT /
        "results/split/train_files.json"

    ) as f:

        train_files = json.load(f)

    # Load test files

    with open(

        PROJECT_ROOT /
        "results/split/test_files.json"

    ) as f:

        test_files = json.load(f)

    # ------------------------------
    # Load Data
    # ------------------------------

    train_df = load_files(

        train_files

    )

    test_df = load_files(

        test_files

    )

    print("\nTrain shape:")

    print(
        train_df.shape
    )

    print("\nTest shape:")

    print(
        test_df.shape
    )

    # ------------------------------
    # Label Mapping
    # ------------------------------

    train_df = map_labels(

        train_df

    )

    test_df = map_labels(

        test_df

    )

    # ------------------------------
    # Cleaning
    # ------------------------------

    train_df = clean_dataset(

        train_df,

        remove_cols=["label"]

    )

    test_df = clean_dataset(

        test_df,

        remove_cols=["label"]

    )

    # ------------------------------
    # Feature Processing
    # ------------------------------

    train_df = process_features(

        train_df

    )

    test_df = process_features(

        test_df

    )

    # ==================================================
    # SELECT BENIGN DATA ONLY
    # ==================================================

    benign_train = train_df[

        train_df["target"] == 0

    ]

    print(

        "\nBenign samples for training:",

        benign_train.shape

    )

    X_benign = benign_train.drop(

        columns=["target"]

    )

    # ==================================================
    # TRAIN ISOLATION FOREST
    # ==================================================

    model = train_isolation_forest(

        X_benign

    )

    # ==================================================
    # TEST ANOMALY DETECTION
    # ==================================================

    X_test = test_df.drop(

        columns=["target"]

    )

    anomaly_result = predict_anomaly(

        model,

        X_test

    )

    anomaly_result["true_label"] = (

        test_df["target"]

        .values

    )

    # Save result

    output_path = (

        PROJECT_ROOT
        /
        "results"
        /
        "anomaly"

    )

    output_path.mkdir(

        parents=True,

        exist_ok=True

    )

    anomaly_result.to_csv(

        output_path /
        "anomaly_scores.csv",

        index=False

    )

    print(

        "\nAnomaly results saved:",

        output_path /
        "anomaly_scores.csv"

    )

    print("\n")

    print("="*70)

    print(
        "Isolation Forest Training Completed Successfully"
    )

    print("="*70)


# ==========================================================
# RUN
# ==========================================================
if __name__ == "__main__":

    main()
