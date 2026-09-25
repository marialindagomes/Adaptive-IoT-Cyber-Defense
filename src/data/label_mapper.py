"""
===========================================================
Label Mapping Module

CICIoT2023:
BenignTraffic -> 0
Attack -> 1


Edge-IIoTset:
Attack_label:
0 -> Normal
1 -> Attack


Output:
target column

0 = Normal
1 = Attack

===========================================================
"""


import pandas as pd


# ==================================================
# CICIoT2023 Label Mapping
# ==================================================

def map_ciciot2023_labels(df):

    if "label" not in df.columns:

        raise ValueError(
            "CICIoT2023 label column not found"
        )

    df = df.copy()

    df["target"] = (

        df["label"]

        .apply(

            lambda x:

            0 if str(x).lower()
            == "benigntraffic"

            else 1

        )

    )

    return df


# ==================================================
# Edge-IIoTset Label Mapping
# ==================================================

def map_edgeiiotset_labels(df):

    if "Attack_label" not in df.columns:

        raise ValueError(
            "Edge-IIoTset Attack_label column not found"
        )

    df = df.copy()

    df["target"] = (

        df["Attack_label"]

        .astype(int)

    )

    return df


# ==================================================
# Verify Mapping
# ==================================================

def check_distribution(df):

    print(
        "\nTarget Distribution:"
    )

    print(

        df["target"]

        .value_counts()

    )
