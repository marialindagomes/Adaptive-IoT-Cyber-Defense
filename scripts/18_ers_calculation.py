"""
===========================================================
ERS Calculation Pipeline

Input:
SHAP values + model prediction

Output:
ERS report

===========================================================
"""


from pathlib import Path

import pandas as pd

import json

import xgboost as xgb

import shap

import importlib.util


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_module(name, path):

    spec = importlib.util.spec_from_file_location(

        name,

        path

    )

    module = importlib.util.module_from_spec(

        spec

    )

    spec.loader.exec_module(

        module

    )

    return module


ers_module = load_module(

    "ers",

    PROJECT_ROOT
    /
    "src"
    /
    "explainability"
    /
    "ers.py"

)


def main():

    print("="*70)

    print(
        "Explanation Reliability Score"
    )

    print("="*70)

    model = xgb.XGBClassifier()

    model.load_model(

        PROJECT_ROOT
        /
        "models"
        /
        "adaptive"
        /
        "adaptive_xgboost.json"

    )

    df = pd.read_csv(

        PROJECT_ROOT
        /
        "data"
        /
        "processed"
        /
        "ciciot2023"
        /
        "test_features.csv",

        nrows=1000

    )

    X = df.drop(

        columns=["target"]

    )

    probability = model.predict_proba(

        X

    )[:, 1]

    explainer = shap.TreeExplainer(

        model

    )

    shap_values = explainer.shap_values(

        X

    )

    result = ers_module.calculate_ers(

        shap_values,

        probability

    )

    output = (

        PROJECT_ROOT
        /
        "results"
        /
        "ers"

    )

    output.mkdir(

        parents=True,

        exist_ok=True

    )

    with open(

        output /
        "ers_report.json",

        "w"

    ) as f:

        json.dump(

            result,

            f,

            indent=4

        )

    print(result)

    print(
        "ERS Completed Successfully"
    )


if __name__ == "__main__":

    main()
