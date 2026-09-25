"""
===========================================================
Test Label Mapping Module

No package import required
===========================================================
"""

from pathlib import Path
import sys
import pandas as pd
import importlib.util


# ==========================================================
# PROJECT PATH
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


LABEL_MAPPER_PATH = (
    PROJECT_ROOT
    /
    "src"
    /
    "data"
    /
    "label_mapper.py"
)


# ==========================================================
# LOAD label_mapper.py DIRECTLY
# ==========================================================

spec = importlib.util.spec_from_file_location(
    "label_mapper",
    LABEL_MAPPER_PATH
)


label_mapper = importlib.util.module_from_spec(
    spec
)


spec.loader.exec_module(
    label_mapper
)


# Get functions

map_ciciot2023_labels = (
    label_mapper.map_ciciot2023_labels
)


map_edgeiiotset_labels = (
    label_mapper.map_edgeiiotset_labels
)


check_distribution = (
    label_mapper.check_distribution
)


# ==========================================================
# CICIoT2023 TEST
# ==========================================================

def test_ciciot2023():

    print("=" * 60)

    print(
        "CICIoT2023 Label Mapping Test"
    )

    print("=" * 60)

    csv_path = (
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

    file = sorted(
        csv_path.glob("*.csv")
    )[0]

    df = pd.read_csv(
        file,
        nrows=10000
    )

    print("\nOriginal labels:")

    print(
        df["label"]
        .value_counts()
    )

    df = map_ciciot2023_labels(
        df
    )

    print("\nAfter mapping:")

    check_distribution(
        df
    )


# ==========================================================
# Edge-IIoTset TEST
# ==========================================================

def test_edgeiiotset():

    print("\n")

    print("=" * 60)

    print(
        "Edge-IIoTset Label Mapping Test"
    )

    print("=" * 60)

    csv_path = (
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

    file = sorted(
        csv_path.glob("*.csv")
    )[0]

    df = pd.read_csv(
        file,
        nrows=10000
    )

    print("\nOriginal labels:")

    print(
        df["Attack_label"]
        .value_counts()
    )

    df = map_edgeiiotset_labels(
        df
    )

    print("\nAfter mapping:")

    check_distribution(
        df
    )


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    test_ciciot2023()

    test_edgeiiotset()

    print("\n")

    print("=" * 60)

    print(
        "Label Mapping Test Completed Successfully"
    )

    print("=" * 60)
