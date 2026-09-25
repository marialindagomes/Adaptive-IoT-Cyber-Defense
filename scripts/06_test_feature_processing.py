"""
===========================================================
Test Feature Processing Pipeline

Purpose:
- Test feature selection
- Test leakage removal
- Prepare ML-ready dataset

===========================================================
"""


from pathlib import Path
import pandas as pd
import importlib.util


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
# LOAD LABEL MAPPER
# ==========================================================
label_mapper = load_module(

    "label_mapper",

    PROJECT_ROOT
    /
    "src"
    /
    "data"
    /
    "label_mapper.py"

)


map_ciciot2023_labels = (

    label_mapper
    .map_ciciot2023_labels

)


# ==========================================================
# LOAD CLEAN MODULE
# ==========================================================

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


clean_dataset = (

    clean_module
    .clean_dataset

)


# ==========================================================
# LOAD FEATURE MODULE
# ==========================================================

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


process_features = (

    feature_module
    .process_features

)


# ==========================================================
# MAIN TEST
# ==========================================================


def test_feature_processing():

    print("="*60)

    print(
        "Feature Processing Test"
    )

    print("="*60)

    # Dataset path

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

    files = sorted(

        data_path.glob("*.csv")

    )

    if not files:

        raise FileNotFoundError(
            "CICIoT2023 CSV not found"
        )

    # Load sample

    df = pd.read_csv(

        files[0],

        nrows=100000

    )

    print("\nOriginal Shape:")

    print(
        df.shape
    )

    # ==========================
    # Label Mapping
    # ==========================

    df = map_ciciot2023_labels(

        df

    )

    print("\nAfter Label Mapping:")

    print(

        df["target"]

        .value_counts()

    )

    # ==========================
    # Cleaning
    # ==========================

    df = clean_dataset(

        df,

        remove_cols=[

            "label"

        ]

    )

    print("\nAfter Cleaning:")

    print(
        df.shape
    )

    # ==========================
    # Feature Processing
    # ==========================

    processed_df = process_features(

        df,

        save_path="results/feature_processing/ciciot_features.json"

    )

    print("\nFinal ML Dataset:")

    print(

        processed_df.shape

    )

    print("\nColumns:")

    print(

        list(processed_df.columns)

    )

    print("\nFeature Processing Test Completed Successfully")


# ==========================================================
# RUN
# ==========================================================
if __name__ == "__main__":

    test_feature_processing()
