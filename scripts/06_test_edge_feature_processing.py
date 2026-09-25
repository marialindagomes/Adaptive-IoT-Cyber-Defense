"""
===========================================================
Edge-IIoTset Feature Processing Test

Purpose:
- Apply label mapping
- Clean dataset
- Remove leakage features
- Select numeric features
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
# LOAD LABEL MAPPER
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


map_edgeiiotset_labels = (

    label_module
    .map_edgeiiotset_labels

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

def test_edge_feature_processing():

    print("="*60)

    print(
        "Edge-IIoTset Feature Processing Test"
    )

    print("="*60)

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

    # Load sample

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
        len(df.columns)
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
    # CLEANING
    # ==================================================

    df = clean_dataset(

        df,

        remove_cols=[

            "Attack_label",

            "Attack_type",

            "frame.time",

            "ip.src_host",

            "ip.dst_host"

        ]

    )

    print("\nAfter Cleaning:")

    print(
        df.shape
    )

    # ==================================================
    # FEATURE PROCESSING
    # ==================================================

    processed_df = process_features(

        df,

        save_path="results/feature_processing/edge_features.json"

    )

    print("\nFinal ML Dataset:")

    print(
        processed_df.shape
    )

    print("\nFinal Columns:")

    print(

        list(processed_df.columns)

    )

    print(
        "\nEdge-IIoTset Feature Processing Completed Successfully"
    )


# ==========================================================
# RUN
# ==========================================================
if __name__ == "__main__":

    test_edge_feature_processing()
