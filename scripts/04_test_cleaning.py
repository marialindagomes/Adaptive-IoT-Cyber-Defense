"""
===========================================================
Test Data Cleaning Pipeline

Purpose:
1. Load CICIoT2023 sample
2. Apply label mapping
3. Apply cleaning
4. Check final output

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

def load_module(module_name, file_path):

    spec = importlib.util.spec_from_file_location(
        module_name,
        file_path
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


# ==========================================================
# LOAD label_mapper.py
# ==========================================================
label_mapper_path = (

    PROJECT_ROOT
    /
    "src"
    /
    "data"
    /
    "label_mapper.py"

)


label_mapper = load_module(
    "label_mapper",
    label_mapper_path
)


map_ciciot2023_labels = (
    label_mapper.map_ciciot2023_labels
)


# ==========================================================
# LOAD clean_data.py
# ==========================================================

clean_data_path = (

    PROJECT_ROOT
    /
    "src"
    /
    "data"
    /
    "clean_data.py"

)


clean_module = load_module(
    "clean_data",
    clean_data_path
)


clean_dataset = (
    clean_module.clean_dataset
)


# ==========================================================
# TEST CLEANING
# ==========================================================

def test_cleaning():

    print("="*60)

    print(
        "CICIoT2023 Cleaning Test"
    )

    print("="*60)

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

    df = pd.read_csv(

        files[0],

        nrows=100000

    )

    print("\nOriginal Shape:")

    print(
        df.shape
    )

    # Label mapping

    df = map_ciciot2023_labels(
        df
    )

    print("\nAfter Label Mapping:")

    print(
        df["target"]
        .value_counts()
    )

    # Cleaning

    df_clean = clean_dataset(

        df,

        remove_cols=[
            "label"
        ]

    )

    print("\nFinal Shape:")

    print(
        df_clean.shape
    )

    print("\nCleaning Test Completed Successfully")


# ==========================================================
# RUN
# ==========================================================
if __name__ == "__main__":

    test_cleaning()
