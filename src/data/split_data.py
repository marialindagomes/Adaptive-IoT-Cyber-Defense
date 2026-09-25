"""
===========================================================
Dataset Split Module

Purpose:
- File-level train/validation/test split
- Prevent data leakage
- Save split information

Dataset:
CICIoT2023

Split:
Train       60%
Validation  20%
Test        20%

===========================================================
"""


from pathlib import Path
import random
import json


# ==========================================================
# GET CSV FILES
# ==========================================================

def get_csv_files(data_path):

    files = sorted(

        Path(data_path)

        .glob("*.csv")

    )

    if not files:

        raise FileNotFoundError(

            "No CSV files found"

        )

    return files


# ==========================================================
# SPLIT FILES
# ==========================================================

def split_files(

        files,

        train_ratio=0.60,

        val_ratio=0.20,

        test_ratio=0.20,

        random_seed=42

):

    if (

        train_ratio

        +

        val_ratio

        +

        test_ratio

        != 1.0

    ):

        raise ValueError(

            "Split ratio must equal 1"

        )

    # Copy list

    files = list(files)

    # Shuffle

    random.seed(
        random_seed
    )

    random.shuffle(
        files
    )

    total = len(files)

    train_end = int(

        total

        *

        train_ratio

    )

    val_end = train_end + int(

        total

        *

        val_ratio

    )

    train_files = files[:train_end]

    val_files = files[train_end:val_end]

    test_files = files[val_end:]

    return (

        train_files,

        val_files,

        test_files

    )


# ==========================================================
# SAVE SPLIT INFORMATION
# ==========================================================

def save_split_files(

        train_files,

        val_files,

        test_files,

        output_path

):

    output_path = Path(
        output_path
    )

    output_path.mkdir(

        parents=True,

        exist_ok=True

    )

    split_data = {


        "train": [

            file.name

            for file in train_files

        ],


        "validation": [

            file.name

            for file in val_files

        ],


        "test": [

            file.name

            for file in test_files

        ]

    }

    with open(

        output_path / "train_files.json",

        "w"

    ) as f:

        json.dump(

            split_data["train"],

            f,

            indent=4

        )

    with open(

        output_path / "validation_files.json",

        "w"

    ) as f:

        json.dump(

            split_data["validation"],

            f,

            indent=4

        )

    with open(

        output_path / "test_files.json",

        "w"

    ) as f:

        json.dump(

            split_data["test"],

            f,

            indent=4

        )

    print(

        "Split files saved at:",

        output_path

    )


# ==========================================================
# MAIN FUNCTION
# ==========================================================

def create_split(

        data_path,

        output_path,

        train_ratio=0.60,

        val_ratio=0.20,

        test_ratio=0.20,

        random_seed=42

):

    print("="*60)

    print(
        "Creating Dataset Split"
    )

    print("="*60)

    files = get_csv_files(

        data_path

    )

    print(

        "Total files:",

        len(files)

    )

    train_files, val_files, test_files = split_files(

        files,

        train_ratio,

        val_ratio,

        test_ratio,

        random_seed

    )

    print("\nSplit Result:")

    print(

        "Train:",

        len(train_files),

        "files"

    )

    print(

        "Validation:",

        len(val_files),

        "files"

    )

    print(

        "Test:",

        len(test_files),

        "files"

    )

    save_split_files(

        train_files,

        val_files,

        test_files,

        output_path

    )

    return (

        train_files,

        val_files,

        test_files

    )
