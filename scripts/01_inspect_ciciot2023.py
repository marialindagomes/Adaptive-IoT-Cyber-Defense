"""
===========================================================
CICIoT2023 Dataset Inspection Pipeline

Purpose:
- Count CSV files
- Inspect dataset structure
- Analyze labels
- Check missing values
- Check duplicate rows
- Generate reports

Dataset:
CICIoT2023

===========================================================
"""


from pathlib import Path
import pandas as pd
import json
try:
    from tqdm import tqdm  # pyright: ignore[reportMissingModuleSource]
except ImportError:
    # Keep the inspection pipeline usable when the optional progress-bar
    # dependency is not installed.
    def tqdm(iterable, *args, **kwargs):
        return iterable


# ==========================================================
# PATH CONFIGURATION
# ==========================================================

DATA_PATH = Path(
    "data/raw/ciciot2023/csv"
)


RESULT_PATH = Path(
    "results/data_inspection"
)


RESULT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# FIND CSV FILES
# ==========================================================

def get_csv_files():

    csv_files = sorted(
        DATA_PATH.glob("*.csv")
    )

    if not csv_files:

        raise FileNotFoundError(
            f"No CSV files found in {DATA_PATH}"
        )

    return csv_files


# ==========================================================
# INSPECT SINGLE CSV FILE
# ==========================================================

def inspect_file(file_path):

    df = pd.read_csv(
        file_path
    )

    rows = int(
        df.shape[0]
    )

    columns = int(
        df.shape[1]
    )

    # Missing values

    missing_values = int(
        df.isnull()
        .sum()
        .sum()
    )

    # Duplicate rows

    duplicate_rows = int(
        df.duplicated()
        .sum()
    )

    result = {


        "file_name":

            file_path.name,


        "rows":

            rows,


        "columns":

            columns,


        "column_names":

            list(df.columns),


        "missing_values":

            missing_values,


        "duplicate_rows":

            duplicate_rows,


        "data_types":

            {

                str(col):

                str(dtype)

                for col, dtype

                in df.dtypes.items()

            }

    }

    # Label analysis

    if "label" in df.columns:

        label_counts = {}

        for label, count in (

            df["label"]

            .value_counts()

            .items()

        ):

            label_counts[str(label)] = int(count)

        result["labels"] = label_counts

    else:

        result["labels"] = {}

    return result


# ==========================================================
# MAIN INSPECTION
# ==========================================================


def inspect_dataset():

    print("=" * 70)

    print(
        "CICIoT2023 Dataset Inspection Started"
    )

    print("=" * 70)

    csv_files = get_csv_files()

    print(
        f"\nTotal CSV files found: {len(csv_files)}"
    )

    dataset_report = []

    total_rows = 0

    global_labels = {}

    # Process all CSV files

    for file in tqdm(

        csv_files,

        desc="Inspecting CSV files"

    ):

        report = inspect_file(
            file
        )

        dataset_report.append(
            report
        )

        total_rows += report["rows"]

        # Merge labels

        for label, count in report["labels"].items():

            if label in global_labels:

                global_labels[label] += count

            else:

                global_labels[label] = count

    # ======================================================
    # SAVE FULL DATASET REPORT
    # ======================================================

    with open(

        RESULT_PATH /

        "ciciot2023_dataset_summary.json",

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            dataset_report,

            file,

            indent=4,

            ensure_ascii=False

        )

    # ======================================================
    # SAVE LABEL DISTRIBUTION
    # ======================================================

    label_df = pd.DataFrame(

        list(
            global_labels.items()
        ),

        columns=[

            "label",

            "count"

        ]

    )

    label_df = label_df.sort_values(

        by="count",

        ascending=False

    )

    label_df.to_csv(

        RESULT_PATH /

        "ciciot2023_label_distribution.csv",

        index=False

    )

    # ======================================================
    # SAVE SUMMARY
    # ======================================================

    summary = {


        "total_csv_files":

            int(len(csv_files)),


        "total_rows":

            int(total_rows),


        "total_classes":

            int(len(global_labels)),


        "labels":

            global_labels

    }

    with open(

        RESULT_PATH /

        "ciciot2023_summary.json",

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            summary,

            file,

            indent=4,

            ensure_ascii=False

        )

    # ======================================================
    # FINAL OUTPUT
    # ======================================================

    print("\n")

    print("=" * 70)

    print(
        "Inspection Completed Successfully"
    )

    print("=" * 70)

    print(
        "\nTotal Files:",
        len(csv_files)
    )

    print(
        "Total Rows:",
        total_rows
    )

    print(
        "Total Classes:",
        len(global_labels)
    )

    print(
        "\nLabel Distribution:"
    )

    for label, count in global_labels.items():

        print(
            f"{label}: {count}"
        )

    print(
        "\nReports saved at:"
    )

    print(
        RESULT_PATH
    )


# ==========================================================
# RUN SCRIPT
# ==========================================================

if __name__ == "__main__":

    inspect_dataset()
