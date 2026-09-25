"""
===========================================================
Data Cleaning Module

Purpose:
- Remove duplicate rows
- Handle infinite values
- Handle missing values
- Remove unwanted columns
- Optimize memory

Used for:
- CICIoT2023
- Edge-IIoTset

===========================================================
"""


import pandas as pd
import numpy as np


# ==========================================================
# REMOVE INFINITE VALUES
# ==========================================================

def handle_infinite_values(df):

    df = df.replace(

        [np.inf, -np.inf],

        np.nan

    )

    return df


# ==========================================================
# REMOVE DUPLICATES
# ==========================================================

def remove_duplicates(df):

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    removed = before - after

    print(
        f"Duplicate rows removed: {removed}"
    )

    return df


# ==========================================================
# HANDLE MISSING VALUES
# ==========================================================

def handle_missing_values(df):

    missing = (

        df.isnull()

        .sum()

        .sum()

    )

    print(
        f"Missing values found: {missing}"
    )

    # For now only report
    # Imputation will be done
    # inside preprocessing pipeline

    return df


# ==========================================================
# REMOVE UNWANTED COLUMNS
# ==========================================================

def remove_columns(df, columns):

    existing_columns = [

        col

        for col in columns

        if col in df.columns

    ]

    if existing_columns:

        df = df.drop(

            columns=existing_columns

        )

        print(

            "Removed columns:",

            existing_columns

        )

    return df


# ==========================================================
# MEMORY OPTIMIZATION
# ==========================================================

def optimize_memory(df):

    for col in df.columns:

        col_type = df[col].dtype

        if col_type == "float64":

            df[col] = df[col].astype(

                "float32"

            )

        elif col_type == "int64":

            df[col] = df[col].astype(

                "int32"

            )

    return df


# ==========================================================
# MAIN CLEAN FUNCTION
# ==========================================================

def clean_dataset(

        df,

        remove_duplicate=True,

        remove_cols=None

):

    print("="*60)

    print(
        "Starting Data Cleaning"
    )

    print("="*60)

    print(

        "Original shape:",

        df.shape

    )

    # 1. Infinite values

    df = handle_infinite_values(
        df
    )

    # 2. Duplicate removal

    if remove_duplicate:

        df = remove_duplicates(
            df
        )

    # 3. Missing values

    df = handle_missing_values(
        df
    )

    # 4. Remove columns

    if remove_cols:

        df = remove_columns(

            df,

            remove_cols

        )

    # 5. Optimize memory

    df = optimize_memory(
        df
    )

    print(

        "Cleaned shape:",

        df.shape

    )

    print("="*60)

    print(
        "Cleaning Completed"
    )

    print("="*60)

    return df
