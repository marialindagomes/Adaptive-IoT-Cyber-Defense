"""
===========================================================
Edge-IIoTset Dataset Inspection Pipeline

Purpose:
- Count CSV files
- Inspect dataset structure
- Analyze labels
- Check missing values
- Check duplicate rows
- Generate reports

Dataset:
Edge-IIoTset

===========================================================
"""


from pathlib import Path
import pandas as pd
import json


def tqdm(iterable, **_kwargs):
    """Fallback progress iterator that requires no external dependency."""
    return iterable


# ==========================================================
# PATH CONFIGURATION
# ==========================================================

DATA_PATH = Path(
    "data/raw/edge_iiotset/csv"
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

    if len(csv_files) == 0:

        raise FileNotFoundError(
            f"No CSV files found in {DATA_PATH}"
        )

    return csv_files


# ==========================================================
# INSPECT SINGLE FILE
# ==========================================================

def inspect_file(file_path):

    df = pd.read_csv(
        file_path
    )

    result = {


        "file_name":

            file_path.name,


        "rows":

            int(df.shape[0]),


        "columns":

            int(df.shape[1]),


        "column_names":

            list(df.columns),


        "missing_values":

            int(
                df.isnull()
                .sum()
                .sum()
            ),


        "duplicate_rows":

            int(
                df.duplicated()
                .sum()
            ),


        "data_types":

            {

                str(col):

                str(dtype)

                for col, dtype

                in df.dtypes.items()

            }

    }

    # Label detection

    possible_labels = [

        "Attack_label",

        "Attack_type",

        "label",

        "Label",

        "attack",

        "Attack",

        "class",

        "Class"

    ]

    label_column = None

    for col in possible_labels:

        if col in df.columns:

            label_column = col

            break

    if label_column:

        label_counts = {}

        for label, count in (

            df[label_column]

            .value_counts()

            .items()

        ):

            label_counts[str(label)] = int(count)

        result["label_column"] = label_column

        result["labels"] = label_counts

    else:

        result["label_column"] = None

        result["labels"] = {}

    return result


# ==========================================================
# MAIN INSPECTION
# ==========================================================

def inspect_dataset():

    print("=" * 70)

    print(
        "Edge-IIoTset Dataset Inspection Started"
    )

    print("=" * 70)

    csv_files = get_csv_files()

    print(
        f"\nTotal CSV files found: {len(csv_files)}"
    )

    dataset_report = []

    total_rows = 0

    global_labels = {}

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

        for label, count in report["labels"].items():

            if label in global_labels:

                global_labels[label] += count

            else:

                global_labels[label] = count

    # ======================================================
    # SAVE FULL REPORT
    # ======================================================

    with open(

        RESULT_PATH /

        "edgeiiotset_dataset_summary.json",

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

        list(global_labels.items()),

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

        "edgeiiotset_label_distribution.csv",

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

        "edgeiiotset_summary.json",

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
# RUN
# ==========================================================
if __name__ == "__main__":

    inspect_dataset()
