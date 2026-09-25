"""
===========================================================
Feature Alignment Module

Purpose:
Convert stream features into
CICIoT2023 XGBoost compatible format

===========================================================
"""


from pathlib import Path

import pandas as pd

import json


# ==========================================================
# LOAD MODEL FEATURES
# ==========================================================

def load_feature_list(

        feature_file

):

    feature_file = Path(

        feature_file

    )

    with open(

        feature_file,

        "r"

    ) as f:

        features = json.load(

            f

        )

    return features


# ==========================================================
# ALIGN FEATURES
# ==========================================================

def align_features(

        df,

        required_features

):

    df = df.copy()

    print("="*60)

    print(
        "Feature Alignment"
    )

    print("="*60)

    current_features = set(

        df.columns

    )

    required_features = list(

        required_features

    )

    # Missing features

    missing = [

        col

        for col in required_features

        if col not in current_features

    ]

    # Extra features

    extra = [

        col

        for col in df.columns

        if col not in required_features

    ]

    print(

        "Current features:",

        len(df.columns)

    )

    print(

        "Required features:",

        len(required_features)

    )

    print(

        "Missing features:",

        len(missing)

    )

    print(

        "Extra features:",

        len(extra)

    )

    # Add missing columns

    for col in missing:

        df[col] = 0

    # Remove extra columns

    if extra:

        df = df.drop(

            columns=extra

        )

    # Reorder

    df = df[

        required_features

    ]

    print(

        "Final aligned shape:",

        df.shape

    )

    return df


# ==========================================================
# SAVE ALIGNED DATA
# ==========================================================

def save_aligned_data(

        df,

        output_path

):

    output_path = Path(

        output_path

    )

    output_path.parent.mkdir(

        parents=True,

        exist_ok=True

    )

    df.to_csv(

        output_path,

        index=False

    )

    print(

        "Aligned dataset saved:",

        output_path

    )
