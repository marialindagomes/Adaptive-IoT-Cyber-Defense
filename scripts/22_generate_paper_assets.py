"""
===========================================================
Paper Result Table and Figure Generator

Generate:
- Publication tables
- Research figures

===========================================================
"""


from pathlib import Path

import pandas as pd

import json

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent


OUTPUT = (

    PROJECT_ROOT

    /

    "paper_assets"

)


TABLE_DIR = OUTPUT / "tables"

FIG_DIR = OUTPUT / "figures"


TABLE_DIR.mkdir(

    parents=True,

    exist_ok=True

)


FIG_DIR.mkdir(

    parents=True,

    exist_ok=True

)


# ==========================================================
# Helper
# ==========================================================

def load_json(path):

    with open(path) as f:

        return json.load(f)


def save_table(data, name):

    df = pd.DataFrame(data)

    df.to_csv(

        TABLE_DIR / name,

        index=False

    )


# ==========================================================
# Table 1 Dataset Summary
# ==========================================================

def dataset_table():

    data = [

        {

            "Dataset": "CICIoT2023",

            "Purpose": "Training, Testing, Drift Evaluation",

            "Features": 46

        },


        {

            "Dataset": "Edge-IIoTset",

            "Purpose": "Cross Dataset Validation",

            "Features": 46

        }

    ]

    save_table(

        data,

        "Table1_dataset_summary.csv"

    )


# ==========================================================
# XGBoost Table
# ==========================================================

def xgboost_table():

    path = (

        PROJECT_ROOT

        /

        "results"

        /

        "xgboost"

        /

        "xgboost_metrics.json"

    )

    if path.exists():

        result = load_json(path)

        data = []

        for k, v in result.items():

            if isinstance(v, (int, float)):

                data.append(

                    {

                        "Metric": k,

                        "Value": v

                    }

                )

        save_table(

            data,

            "Table3_xgboost_results.csv"

        )


# ==========================================================
# Calibration
# ==========================================================

def calibration_table():

    path = (

        PROJECT_ROOT

        /

        "results"

        /

        "calibration"

        /

        "calibration_metrics.json"

    )

    if path.exists():

        data = load_json(path)

        rows = []

        for method, values in data.items():

            for metric, value in values.items():

                rows.append(

                    {

                        "Method": method,

                        "Metric": metric,

                        "Value": value

                    }

                )

        save_table(

            rows,

            "Table4_calibration_results.csv"

        )


# ==========================================================
# SHAP
# ==========================================================

def shap_table():

    path = (

        PROJECT_ROOT

        /

        "results"

        /

        "shap"

        /

        "feature_importance.csv"

    )

    if path.exists():

        df = pd.read_csv(path)

        df.head(20).to_csv(

            TABLE_DIR /

            "Table7_shap_features.csv",

            index=False

        )

        plt.figure(figsize=(8, 6))

        temp = df.head(10)

        plt.barh(

            temp["feature"][::-1],

            temp["importance"][::-1]

        )

        plt.xlabel(

            "SHAP Importance"

        )

        plt.title(

            "Top SHAP Features"

        )

        plt.tight_layout()

        plt.savefig(

            FIG_DIR /

            "Figure4_SHAP.png",

            dpi=300

        )

        plt.close()


# ==========================================================
# ERS
# ==========================================================

def ers_table():

    path = (

        PROJECT_ROOT

        /

        "results"

        /

        "ers"

        /

        "ers_report.json"

    )

    if path.exists():

        data = load_json(path)

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

            "Table8_ers_results.csv"

        )

        plt.figure(figsize=(6, 4))

        plt.bar(

            list(data.keys()),

            list(data.values())

        )

        plt.xticks(rotation=45)

        plt.title(

            "Explanation Reliability Score"

        )

        plt.tight_layout()

        plt.savefig(

            FIG_DIR /

            "Figure7_ERS.png",

            dpi=300

        )

        plt.close()


# ==========================================================
# Risk
# ==========================================================

def risk_table():

    path = (

        PROJECT_ROOT

        /

        "results"

        /

        "risk"

        /

        "final_risk_report.json"

    )

    if path.exists():

        data = load_json(path)

        rows = []

        for k, v in data["risk_distribution"].items():

            rows.append(

                {

                    "Threat_Level": k,

                    "Count": v

                }

            )

        save_table(

            rows,

            "Table9_risk_distribution.csv"

        )

        plt.figure(figsize=(6, 4))

        plt.bar(

            data["risk_distribution"].keys(),

            data["risk_distribution"].values()

        )

        plt.title(

            "Risk Distribution"

        )

        plt.tight_layout()

        plt.savefig(

            FIG_DIR /

            "Figure8_Risk_Distribution.png",

            dpi=300

        )

        plt.close()


# ==========================================================
# Response
# ==========================================================

def response_table():

    path = (

        PROJECT_ROOT

        /

        "results"

        /

        "response"

        /

        "response_report.json"

    )

    if path.exists():

        data = load_json(path)

        rows = []

        for k, v in data["actions"].items():

            rows.append(

                {

                    "Response": k,

                    "Count": v

                }

            )

        save_table(

            rows,

            "Table9_ERGAR_Response.csv"

        )

        plt.figure(figsize=(6, 4))

        plt.bar(

            data["actions"].keys(),

            data["actions"].values()

        )

        plt.xticks(rotation=30)

        plt.title(

            "ERGAR Response Distribution"

        )

        plt.tight_layout()

        plt.savefig(

            FIG_DIR /

            "Figure9_ERGAR_Response.png",

            dpi=300

        )

        plt.close()


# ==========================================================
# Drift
# ==========================================================

def drift_table():

    path = (

        PROJECT_ROOT

        /

        "results"

        /

        "drift"

        /

        "final_supervised_drift_report.json"

    )

    if path.exists():

        data = load_json(path)

        save_table(

            [

                {

                    "Metric": "Total Samples",

                    "Value": data["total_samples"]

                },


                {

                    "Metric": "Drift Count",

                    "Value": data["drift_count"]

                },


                {

                    "Metric": "Drift Point",

                    "Value": str(data["drift_points"])

                }

            ],

            "Table6_drift_adaptation_results.csv"

        )


# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":

    dataset_table()

    xgboost_table()

    calibration_table()

    shap_table()

    ers_table()

    risk_table()

    response_table()

    drift_table()

    print("="*60)

    print(

        "Paper Assets Generated Successfully"

    )

    print(

        OUTPUT

    )

    print("="*60)
