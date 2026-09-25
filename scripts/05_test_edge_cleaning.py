

"""
===========================================================
Edge-IIoTset Cleaning Test Pipeline

Workflow:

Edge-IIoTset CSV
        |
        ↓
Label Mapping
        |
        ↓
Data Cleaning
        |
        ↓
Remove Leakage Columns
        |
        ↓
Clean Dataset

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
# LOAD PYTHON MODULE DIRECTLY
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


map_edgeiiotset_labels = (

    label_mapper
    .map_edgeiiotset_labels

)


# ==========================================================
# LOAD CLEANING MODULE
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
# MAIN TEST FUNCTION
# ==========================================================


def test_edge_cleaning():

    print("="*60)

    print(
        "Edge-IIoTset Cleaning Test"
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
        "edge_iiotset"
        /
        "csv"

    )

    files = sorted(

        data_path.glob("*.csv")

    )

    if not files:

        raise FileNotFoundError(
            "Edge-IIoTset CSV not found"
        )

    # Load one sample file

    df = pd.read_csv(

        files[0],

        nrows=100000

    )

    print("\nOriginal Shape:")

    print(
        df.shape
    )

    print("\nOriginal Columns:")

    print(
        list(df.columns)
    )

    # ==================================================
    # LABEL MAPPING
    # ==================================================

    df = map_edgeiiotset_labels(

        df

    )

    print("\nAfter Label Mapping:")

    print(

        df["target"]

        .value_counts()

    )

    # ==================================================
    # REMOVE LEAKAGE COLUMNS
    # ==================================================

    remove_columns = [

        "Attack_label",

        "Attack_type",

        "frame.time",

        "ip.src_host",

        "ip.dst_host"

    ]

    # ==================================================
    # CLEAN DATA
    # ==================================================

    df_clean = clean_dataset(

        df,

        remove_cols=remove_columns

    )

    print("\nFinal Shape:")

    print(
        df_clean.shape
    )

    print("\nFinal Columns:")

    print(
        list(df_clean.columns)
    )

    print("\nEdge-IIoTset Cleaning Test Completed Successfully")


# ==========================================================
# RUN
# ==========================================================
if __name__ == "__main__":

    test_edge_cleaning()
