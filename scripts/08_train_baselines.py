"""
===========================================================
Baseline Model Training Pipeline

Models:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. MLP

Dataset:
CICIoT2023

===========================================================
"""


from pathlib import Path
import pandas as pd
import importlib.util
import json


# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==========================================================
# LOAD MODULE FUNCTION
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


model_module = load_module(

    "baseline_models",

    PROJECT_ROOT
    /
    "src"
    /
    "models"
    /
    "baseline_models.py"

)


eval_module = load_module(

    "evaluate",

    PROJECT_ROOT
    /
    "src"
    /
    "models"
    /
    "evaluate.py"

)


# Functions

map_labels = (
    label_module.map_ciciot2023_labels
)


clean_dataset = (
    clean_module.clean_dataset
)


process_features = (
    feature_module.process_features
)


get_models = (
    model_module.get_all_baseline_models
)


evaluate_model = (
    eval_module.evaluate_model
)


# ==========================================================
# LOAD DATA
# ==========================================================

def load_dataset(file_list, sample_rows=50000):

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

    print("\nLoading files...")

    for file_name in file_list:

        file_path = data_path / file_name

        print(
            "Reading:",
            file_name
        )

        df = pd.read_csv(

            file_path,

            nrows=sample_rows

        )

        dfs.append(df)

    data = pd.concat(

        dfs,

        ignore_index=True

    )

    return data


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("="*70)

    print(
        "CICIoT2023 Baseline Training"
    )

    print("="*70)

    # ----------------------------
    # Load split files
    # ----------------------------

    with open(

        PROJECT_ROOT
        /
        "results"
        /
        "split"
        /
        "train_files.json"

    ) as f:

        train_files = json.load(f)

    with open(

        PROJECT_ROOT
        /
        "results"
        /
        "split"
        /
        "test_files.json"

    ) as f:

        test_files = json.load(f)

    # ----------------------------
    # Load data
    # ----------------------------

    train_df = load_dataset(

        train_files

    )

    test_df = load_dataset(

        test_files

    )

    print(
        "\nTrain shape:",
        train_df.shape
    )

    print(
        "Test shape:",
        test_df.shape
    )

    # ----------------------------
    # Label Mapping
    # ----------------------------

    train_df = map_labels(

        train_df

    )

    test_df = map_labels(

        test_df

    )

    # ----------------------------
    # Cleaning
    # ----------------------------

    train_df = clean_dataset(

        train_df,

        remove_cols=["label"]

    )

    test_df = clean_dataset(

        test_df,

        remove_cols=["label"]

    )

    # ----------------------------
    # Feature Processing
    # ----------------------------

    train_df = process_features(

        train_df

    )

    test_df = process_features(

        test_df

    )

    # ----------------------------
    # Split X/y
    # ----------------------------

    X_train = train_df.drop(

        columns=["target"]

    )

    y_train = train_df["target"]

    X_test = test_df.drop(

        columns=["target"]

    )

    y_test = test_df["target"]

    print(

        "\nFinal Training Data:",

        X_train.shape

    )

    print(

        "Final Test Data:",

        X_test.shape

    )

    # ======================================================
    # Train Models
    # ======================================================

    models = get_models()

    result_path = (

        PROJECT_ROOT
        /
        "results"
        /
        "baseline"

    )

    result_path.mkdir(

        parents=True,

        exist_ok=True

    )

    for name, model in models.items():

        print("\n")

        print("="*60)

        print(
            "Training:",
            name
        )

        print("="*60)

        model.fit(

            X_train,

            y_train

        )

        safe_name = (

            name

            .lower()

            .replace(" ", "_")

        )

        evaluate_model(

            model,

            X_test,

            y_test,

            name,

            result_path
            /
            f"{safe_name}.json"

        )

    print("\nBaseline Training Completed Successfully")


if __name__ == "__main__":

    main()
