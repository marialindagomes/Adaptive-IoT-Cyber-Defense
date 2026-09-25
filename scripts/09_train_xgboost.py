"""
===========================================================
XGBoost Training Pipeline

Main Threat Detection Model

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
    "src" /
    "data" /
    "label_mapper.py"
)


clean_module = load_module(
    "clean_data",
    PROJECT_ROOT /
    "src" /
    "data" /
    "clean_data.py"
)


feature_module = load_module(
    "feature_processing",
    PROJECT_ROOT /
    "src" /
    "data" /
    "feature_processing.py"
)


xgb_module = load_module(
    "train_xgboost",
    PROJECT_ROOT /
    "src" /
    "models" /
    "train_xgboost.py"
)


eval_module = load_module(
    "evaluate",
    PROJECT_ROOT /
    "src" /
    "models" /
    "evaluate.py"
)


# Functions

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


train_xgboost = (
    xgb_module
    .train_xgboost
)


save_feature_importance = (
    xgb_module
    .save_feature_importance
)


evaluate_model = (
    eval_module
    .evaluate_model
)


# ==========================================================
# LOAD DATA FUNCTION
# ==========================================================

def load_files(file_names, sample_rows=50000):

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

    print("\nLoading files...")

    for file_name in file_names:

        print(
            "Reading:",
            file_name
        )

        df = pd.read_csv(

            data_path / file_name,

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
        "CICIoT2023 XGBoost Training"
    )

    print("="*70)

    # -----------------------------------
    # Load split information
    # -----------------------------------

    with open(

        PROJECT_ROOT /
        "results" /
        "split" /
        "train_files.json"

    ) as f:

        train_files = json.load(f)

    with open(

        PROJECT_ROOT /
        "results" /
        "split" /
        "validation_files.json"

    ) as f:

        val_files = json.load(f)

    with open(

        PROJECT_ROOT /
        "results" /
        "split" /
        "test_files.json"

    ) as f:

        test_files = json.load(f)

    # -----------------------------------
    # Load data
    # -----------------------------------

    train_df = load_files(
        train_files
    )

    val_df = load_files(
        val_files
    )

    test_df = load_files(
        test_files
    )

    print("\nTrain:")
    print(train_df.shape)

    print("Validation:")
    print(val_df.shape)

    print("Test:")
    print(test_df.shape)

    # -----------------------------------
    # Label Mapping
    # -----------------------------------

    train_df = map_labels(train_df)

    val_df = map_labels(val_df)

    test_df = map_labels(test_df)

    # -----------------------------------
    # Cleaning
    # -----------------------------------

    train_df = clean_dataset(
        train_df,
        remove_cols=["label"]
    )

    val_df = clean_dataset(
        val_df,
        remove_cols=["label"]
    )

    test_df = clean_dataset(
        test_df,
        remove_cols=["label"]
    )

    # -----------------------------------
    # Feature Processing
    # -----------------------------------

    train_df = process_features(
        train_df
    )

    val_df = process_features(
        val_df
    )

    test_df = process_features(
        test_df
    )

    # -----------------------------------
    # Split X/y
    # -----------------------------------

    X_train = train_df.drop(
        columns=["target"]
    )

    y_train = train_df["target"]

    X_val = val_df.drop(
        columns=["target"]
    )

    y_val = val_df["target"]

    X_test = test_df.drop(
        columns=["target"]
    )

    y_test = test_df["target"]

    print("\nFinal Shapes:")

    print(
        "Train:",
        X_train.shape
    )

    print(
        "Validation:",
        X_val.shape
    )

    print(
        "Test:",
        X_test.shape
    )

    # -----------------------------------
    # Train XGBoost
    # -----------------------------------

    model = train_xgboost(

        X_train,

        y_train,

        X_val,

        y_val

    )

    # -----------------------------------
    # Evaluate
    # -----------------------------------

    results = evaluate_model(

        model,

        X_test,

        y_test,

        "XGBoost",

        "results/xgboost/xgboost_metrics.json"

    )

    # -----------------------------------
    # Feature Importance
    # -----------------------------------

    save_feature_importance(

        model,

        X_train.columns,

        "results/xgboost/feature_importance.csv"

    )

    print("\n")

    print("="*70)

    print(
        "XGBoost Training Completed Successfully"
    )

    print("="*70)

    print(results)


# ==========================================================
# RUN
# ==========================================================
if __name__ == "__main__":

    main()
