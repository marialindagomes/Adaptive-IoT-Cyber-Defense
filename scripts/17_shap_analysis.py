"""
===========================================================
SHAP Analysis Pipeline

Input:
Adaptive XGBoost Model

Output:
SHAP values and feature importance

===========================================================
"""


from pathlib import Path

import pandas as pd

import joblib

import json

import importlib.util


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==========================================================
# LOAD MODULE
# ==========================================================

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


shap_module = load_module(

    "shap_explainer",

    PROJECT_ROOT
    /
    "src"
    /
    "explainability"
    /
    "shap_explainer.py"

)


calculate_shap_values = (

    shap_module

    .calculate_shap_values

)


feature_importance = (

    shap_module

    .feature_importance

)


def main():

    print("="*70)

    print(
        "SHAP Explainability Analysis"
    )

    print("="*70)

    # ------------------------------------------------------
    # Load adaptive model
    # ------------------------------------------------------

    model_path = (

        PROJECT_ROOT
        /
        "models"
        /
        "adaptive"
        /
        "adaptive_xgboost.json"

    )

    import xgboost as xgb

    model = xgb.XGBClassifier()

    model.load_model(

        model_path

    )

    print(

        "Adaptive model loaded"

    )

    # ------------------------------------------------------
    # Load sample data
    # ------------------------------------------------------

    data_file = (

        PROJECT_ROOT
        /
        "data"
        /
        "processed"
        /
        "ciciot2023"
        /
        "test_features.csv"

    )

    df = pd.read_csv(

        data_file,

        nrows=5000

    )

    X = df.drop(

        columns=["target"]

    )

    print(

        "Samples:",

        len(X)

    )

    # ------------------------------------------------------
    # SHAP
    # ------------------------------------------------------

    print(

        "Calculating SHAP values..."

    )

    shap_values, explainer = calculate_shap_values(

        model,

        X

    )

    importance = feature_importance(

        shap_values,

        X.columns

    )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    output = (

        PROJECT_ROOT
        /
        "results"
        /
        "shap"

    )

    output.mkdir(

        parents=True,

        exist_ok=True

    )

    importance.to_csv(

        output /
        "feature_importance.csv",

        index=False

    )

    with open(

        output /
        "shap_report.json",

        "w"

    ) as f:

        json.dump(

            {

                "samples":

                    len(X),


                "top_features":

                    importance.head(20)

                    .to_dict(

                        orient="records"

                    )

            },

            f,

            indent=4

        )

    print()

    print(

        "Saved:",

        output

    )

    print(

        "SHAP Analysis Completed Successfully"

    )


if __name__ == "__main__":

    main()
