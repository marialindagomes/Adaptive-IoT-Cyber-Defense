"""
===========================================================
Save Processed Edge-IIoTset Dataset

Memory Optimized Version

===========================================================
"""


from pathlib import Path
import importlib.util

import pandas as pd


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


label_module = load_module(
    "label_mapper",
    PROJECT_ROOT / "src/data/label_mapper.py"
)


clean_module = load_module(
    "clean_data",
    PROJECT_ROOT / "src/data/clean_data.py"
)


feature_module = load_module(
    "feature_processing",
    PROJECT_ROOT / "src/data/feature_processing.py"
)


map_labels = (
    label_module.map_edgeiiotset_labels
)


clean_dataset = (
    clean_module.clean_dataset
)


process_features = (
    feature_module.process_features
)


# ==========================================================
# PROCESS SINGLE FILE
# ==========================================================

def process_single_file(file_path):

    print("\nProcessing:")
    print(file_path.name)

    df = pd.read_csv(

        file_path,

        low_memory=False

    )

    print(
        "Original:",
        df.shape
    )

    # Label mapping

    df = map_labels(df)

    # Cleaning

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

    # Feature processing

    df = process_features(df)

    print(
        "Processed:",
        df.shape
    )

    return df


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("="*70)

    print(
        "Saving Processed Edge-IIoTset Dataset"
    )

    print("="*70)

    input_path = (

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

        input_path.glob("*.csv")

    )

    print(

        "Total files:",

        len(files)

    )

    processed_path = (

        PROJECT_ROOT
        /
        "data"
        /
        "processed"
        /
        "edge_iiotset"

    )

    processed_path.mkdir(

        parents=True,

        exist_ok=True

    )

    output_file = (

        processed_path
        /
        "features.csv"

    )

    first_write = True

    for file in files:

        processed_df = process_single_file(

            file

        )

        # append mode

        processed_df.to_csv(

            output_file,

            mode="w" if first_write else "a",

            header=first_write,

            index=False

        )

        first_write = False

        print(

            "Saved chunk:",

            file.name

        )

        # release memory

        del processed_df

    print("\n")

    print("="*70)

    print(
        "Edge-IIoTset Processed Dataset Saved Successfully"
    )

    print("="*70)

    print(

        "Output:",

        output_file

    )


if __name__ == "__main__":

    main()
