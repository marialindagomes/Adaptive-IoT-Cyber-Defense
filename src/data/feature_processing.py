"""
===========================================================
Feature Processing Module

Purpose:
- Select usable ML features
- Remove leakage columns
- Handle non-numeric columns
- Prepare data for ML models

Used for:
- CICIoT2023
- Edge-IIoTset

===========================================================
"""


from pathlib import Path
import pandas as pd
import json


# ==========================================================
# REMOVE LEAKAGE COLUMNS
# ==========================================================

def remove_leakage_columns(df):

    leakage_columns = [

        # Labels

        "label",

        "Attack_label",

        "Attack_type",


        # Network identifiers

        "source_file",

        "frame.time",

        "ip.src_host",

        "ip.dst_host"

    ]

    existing = [

        col

        for col in leakage_columns

        if col in df.columns

    ]

    if existing:

        df = df.drop(

            columns=existing

        )

        print(

            "Removed leakage columns:",

            existing

        )

    return df


# ==========================================================
# SELECT NUMERIC FEATURES
# ==========================================================

def select_numeric_features(df):

    numeric_df = (

        df

        .select_dtypes(

            include=[

                "int32",

                "int64",

                "float32",

                "float64"

            ]

        )

    )

    print(

        "Numeric features:",

        numeric_df.shape[1]

    )

    return numeric_df


# ==========================================================
# SPLIT FEATURES AND TARGET
# ==========================================================

def split_features_target(df):

    if "target" not in df.columns:

        raise ValueError(

            "target column not found"

        )

    X = df.drop(

        columns=[

            "target"

        ]

    )

    y = df["target"]

    return X, y


# ==========================================================
# SAVE FEATURE LIST
# ==========================================================

def save_feature_list(

        features,

        output_path

):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(

        parents=True,

        exist_ok=True

    )

    with open(

        output_path,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            list(features),

            file,

            indent=4

        )

    print(

        "Feature list saved:",

        output_path

    )


# ==========================================================
# MAIN FEATURE PROCESSING FUNCTION
# ==========================================================

def process_features(

        df,

        save_path=None

):

    print("="*60)

    print(
        "Feature Processing Started"
    )

    print("="*60)

    print(

        "Original shape:",

        df.shape

    )

    # Remove leakage

    df = remove_leakage_columns(

        df

    )

    # Keep numerical features

    numeric_df = select_numeric_features(

        df

    )

    # Add target back

    if "target" in df.columns:

        numeric_df["target"] = df["target"]

    print(

        "Processed shape:",

        numeric_df.shape

    )

    # Save feature names

    if save_path:

        features = (

            numeric_df

            .drop(

                columns=["target"],

                errors="ignore"

            )

            .columns

        )

        save_feature_list(

            features,

            save_path

        )

    print("="*60)

    print(
        "Feature Processing Completed"
    )

    print("="*60)

    return numeric_df
