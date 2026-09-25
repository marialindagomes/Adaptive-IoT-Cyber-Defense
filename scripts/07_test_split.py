"""
===========================================================
Test Dataset Split Pipeline

Purpose:
- Test file-level train/validation/test split
- Verify no file overlap
- Save split information

Dataset:
CICIoT2023

===========================================================
"""


from pathlib import Path
import importlib.util
import json


# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==========================================================
# LOAD SPLIT MODULE
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


split_module = load_module(

    "split_data",

    PROJECT_ROOT
    /
    "src"
    /
    "data"
    /
    "split_data.py"

)


create_split = (

    split_module

    .create_split

)


# ==========================================================
# MAIN TEST
# ==========================================================

def test_split():

    print("="*60)

    print(
        "CICIoT2023 Train Validation Test Split"
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

    output_path = (

        PROJECT_ROOT
        /
        "results"
        /
        "split"

    )

    # Create split

    train_files, val_files, test_files = create_split(

        data_path,

        output_path,

        train_ratio=0.60,

        val_ratio=0.20,

        test_ratio=0.20,

        random_seed=42

    )

    # ======================================================
    # CHECK OVERLAP
    # ======================================================

    train_set = set(

        file.name

        for file in train_files

    )

    val_set = set(

        file.name

        for file in val_files

    )

    test_set = set(

        file.name

        for file in test_files

    )

    print("\nChecking overlap...")

    print(

        "Train-Val overlap:",

        len(train_set.intersection(val_set))

    )

    print(

        "Train-Test overlap:",

        len(train_set.intersection(test_set))

    )

    print(

        "Val-Test overlap:",

        len(val_set.intersection(test_set))

    )

    # ======================================================
    # SUMMARY
    # ======================================================

    summary = {


        "total_files":

            len(train_files)
            +
            len(val_files)
            +
            len(test_files),


        "train_files":

            len(train_files),


        "validation_files":

            len(val_files),


        "test_files":

            len(test_files),


        "overlap":

            {

                "train_val": len(
                    train_set.intersection(val_set)
                ),

                "train_test": len(
                    train_set.intersection(test_set)
                ),

                "val_test": len(
                    val_set.intersection(test_set)
                )

            }

    }

    with open(

        output_path
        /
        "split_summary.json",

        "w"

    ) as file:

        json.dump(

            summary,

            file,

            indent=4

        )

    print("\n")

    print("="*60)

    print(
        "Split Test Completed Successfully"
    )

    print("="*60)

    print("\nSummary:")

    print(summary)


# ==========================================================
# RUN
# ==========================================================
if __name__ == "__main__":

    test_split()
