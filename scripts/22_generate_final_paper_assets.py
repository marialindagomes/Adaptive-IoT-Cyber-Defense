"""
===========================================================
AdaptiveExplain-IDS
Final Paper Table and Figure Generator

Uses ONLY existing project results.

===========================================================
"""

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent


OUT = ROOT / "paper_assets_final"

TABLES = OUT / "tables"
FIGURES = OUT / "figures"

TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


def load_json(path):

    with open(path) as f:
        return json.load(f)


def save_table(data, name):

    pd.DataFrame(data).to_csv(
        TABLES / name,
        index=False
    )

    print("Table:", name)


def save_bar(labels, values, title, filename):

    plt.figure(figsize=(8, 5))

    plt.bar(
        labels,
        values
    )

    plt.title(title)

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES / filename,
        dpi=300
    )

    plt.close()

    print("Figure:", filename)


# ======================================================
# Table 1 Dataset Summary
# ======================================================

def dataset_summary():

    data = [

        {
            "Dataset": "CICIoT2023",
            "Purpose": "Training and Testing",
            "Feature_Count": 46
        },

        {
            "Dataset": "Edge-IIoTset",
            "Purpose": "Cross Dataset Validation",
            "Feature_Count": 46
        }

    ]

    save_table(
        data,
        "Table1_Dataset_Summary.csv"
    )


# ======================================================
# Table 3 Baseline Comparison
# ======================================================

def baseline():

    folder = ROOT/"results"/"baseline"

    rows = []

    for f in folder.glob("*.json"):

        data = load_json(f)

        row = {

            "Model":
            f.stem

        }

        for k, v in data.items():

            if isinstance(v, (int, float)):

                row[k] = v

        rows.append(row)

    # XGBoost add

    xgb = load_json(
        ROOT /
        "results" /
        "xgboost" /
        "xgboost_metrics.json"
    )

    row = {"Model": "XGBoost"}

    for k, v in xgb.items():

        if isinstance(v, (int, float)):

            row[k] = v

    rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(
        TABLES /
        "Table3_Baseline_Comparison.csv",
        index=False
    )

    metric = "f1_score"

    if metric not in df.columns:
        metric = "accuracy"

    save_bar(

        df["Model"],

        df[metric],

        "Baseline Model Comparison",

        "Figure2_Baseline_Comparison.png"

    )


# ======================================================
# XGBoost Performance
# ======================================================

def xgboost_result():

    data = load_json(
        ROOT /
        "results" /
        "xgboost" /
        "xgboost_metrics.json"
    )

    rows = []

    for k, v in data.items():

        if isinstance(v, (int, float)):

            rows.append(
                {
                    "Metric": k,
                    "Value": v
                }
            )

    save_table(
        rows,
        "Table4_XGBoost_Performance.csv"
    )

    save_bar(

        [x["Metric"] for x in rows],

        [x["Value"] for x in rows],

        "XGBoost Performance",

        "Figure3_XGBoost_Performance.png"

    )


# ======================================================
# Calibration
# ======================================================

def calibration():

    data = load_json(
        ROOT /
        "results" /
        "calibration" /
        "calibration_metrics.json"
    )

    rows = []

    for method, val in data.items():

        if isinstance(val, dict):

            for k, v in val.items():

                rows.append(
                    {
                        "Method": method,
                        "Metric": k,
                        "Value": v
                    }
                )

    save_table(
        rows,
        "Table5_Calibration.csv"
    )


# ======================================================
# Unknown Attack
# ======================================================

def unknown_attack():

    df = pd.read_csv(

        ROOT /
        "results" /
        "unseen_attack" /
        "unseen_attack_summary.csv"

    )

    df.to_csv(

        TABLES /
        "Table6_Unknown_Attack.csv",

        index=False

    )

    save_bar(

        df.iloc[:, 0],

        df.iloc[:, 1],

        "Unknown Attack Detection",

        "Figure5_Unknown_Attack.png"

    )


# ======================================================
# Drift
# ======================================================

def drift():

    data = load_json(

        ROOT /
        "results" /
        "drift" /
        "final_supervised_drift_report.json"

    )

    rows = []

    for k, v in data.items():

        if isinstance(v, (int, float, str, list)):

            rows.append(

                {
                    "Parameter": k,
                    "Value": str(v)
                }

            )

    save_table(

        rows,

        "Table7_Drift_Adaptive.csv"

    )

    csv_files = list(

        (ROOT /
         "results" /
         "drift").glob("*.csv")

    )

    if not csv_files:

        print(
            "No drift CSV found, skipping figure"
        )

        return

    df = pd.read_csv(

        csv_files[0],

        low_memory=False

    )

    numeric_cols = df.select_dtypes(

        include=["number"]

    ).columns

    if len(numeric_cols) == 0:

        print(
            "No numeric drift column found, creating summary figure"
        )

        drift_count = data.get(

            "drift_count",

            0

        )

        plt.figure(figsize=(6, 4))

        plt.bar(

            ["Detected Drift"],

            [drift_count]

        )

        plt.title(

            "ADWIN Drift Detection Result"

        )

        plt.ylabel(

            "Count"

        )

        plt.tight_layout()

        plt.savefig(

            FIGURES /
            "Figure6_ADWIN_Drift.png",

            dpi=300

        )

        plt.close()

        return

    col = numeric_cols[0]

    plt.figure(figsize=(9, 4))

    plt.plot(

        df[col]

    )

    plt.title(

        "ADWIN Drift Detection"

    )

    plt.xlabel(

        "Stream Index"

    )

    plt.ylabel(

        col

    )

    plt.tight_layout()

    plt.savefig(

        FIGURES /
        "Figure6_ADWIN_Drift.png",

        dpi=300

    )

    plt.close()

    print(

        "Figure6_ADWIN_Drift.png created"

    )
# ======================================================
# SHAP
# ======================================================


def shap():

    df = pd.read_csv(

        ROOT /
        "results" /
        "shap" /
        "feature_importance.csv"

    )

    df.head(10).to_csv(

        TABLES /
        "Table8_SHAP.csv",

        index=False

    )

    temp = df.head(10)

    save_bar(

        temp["feature"],

        temp["importance"],

        "SHAP Feature Importance",

        "Figure8_SHAP.png"

    )


# ======================================================
# ERS
# ======================================================

def ers():

    data = load_json(

        ROOT /
        "results" /
        "ers" /
        "ers_report.json"

    )

    rows = []

    for k, v in data.items():

        rows.append(

            {
                "Component": k,
                "Score": v
            }

        )

    save_table(
        rows,
        "Table9_ERS.csv"
    )

    save_bar(

        list(data.keys()),

        list(data.values()),

        "Explanation Reliability Score",

        "Figure9_ERS.png"

    )


# ======================================================
# Risk
# ======================================================

def risk():

    data = load_json(

        ROOT /
        "results" /
        "risk" /
        "final_risk_report.json"

    )

    rows = []

    for k, v in data["risk_distribution"].items():

        rows.append(
            {
                "Risk_Level": k,
                "Count": v
            }
        )

    save_table(
        rows,
        "Table10_Risk.csv"
    )

    save_bar(

        [x["Risk_Level"] for x in rows],

        [x["Count"] for x in rows],

        "Risk Distribution",

        "Figure10_Risk.png"

    )


# ======================================================
# ERGAR
# ======================================================

def response():

    data = load_json(

        ROOT /
        "results" /
        "response" /
        "response_report.json"

    )

    rows = []

    for k, v in data["actions"].items():

        rows.append(

            {
                "Action": k,
                "Count": v
            }

        )

    save_table(
        rows,
        "Table11_ERGAR.csv"
    )

    save_bar(

        [x["Action"] for x in rows],

        [x["Count"] for x in rows],

        "ERGAR Response",

        "Figure11_ERGAR.png"

    )


# ======================================================
# Edge Validation
# ======================================================

def edge():

    data = load_json(

        ROOT /
        "results" /
        "edge_validation" /
        "edge_metrics.json"

    )

    rows = []

    for k, v in data.items():

        if isinstance(v, (int, float)):

            rows.append(

                {
                    "Metric": k,
                    "Value": v
                }

            )

    save_table(
        rows,
        "Table12_Edge_Validation.csv"
    )

    save_bar(

        [x["Metric"] for x in rows],

        [x["Value"] for x in rows],

        "Edge-IIoTset Validation",

        "Figure12_Edge_Validation.png"

    )


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    print("Generating Final Paper Assets")

    dataset_summary()

    baseline()

    xgboost_result()

    calibration()

    unknown_attack()

    drift()

    shap()

    ers()

    risk()

    response()

    edge()

    print("\nDONE")
    print(OUT)
