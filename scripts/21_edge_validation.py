"""
===========================================================
Edge-IIoTset Cross Dataset Validation FINAL

Input:
validation_ready.csv

Model:
Adaptive XGBoost trained on CICIoT2023

Output:
Edge validation metrics

===========================================================
"""


from pathlib import Path

import pandas as pd

import json

import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():

    print("="*70)

    print(
        "Edge-IIoTset Cross Dataset Validation"
    )

    print("="*70)

    # ======================================================
    # Load Model
    # ======================================================

    model_path = (

        PROJECT_ROOT
        /
        "models"
        /
        "adaptive"
        /
        "adaptive_xgboost.json"

    )

    model = xgb.XGBClassifier()

    model.load_model(

        model_path

    )

    print(

        "Adaptive XGBoost loaded"

    )

    # ======================================================
    # Load Prepared Edge Dataset
    # ======================================================

    data_file = (

        PROJECT_ROOT
        /
        "data"
        /
        "processed"
        /
        "edge_iiotset"
        /
        "validation_ready.csv"

    )

    print(

        "Loading validation dataset..."

    )

    df = pd.read_csv(

        data_file,

        low_memory=False

    )

    print(

        "Samples:",

        len(df)

    )

    # ======================================================
    # Target
    # ======================================================

    if "target" not in df.columns:

        raise ValueError(

            "target column missing"

        )

    y = df["target"].astype(int)

    X = df.drop(

        columns=["target"]

    )

    # ======================================================
    # Feature Alignment
    # ======================================================

    X = X.apply(

        pd.to_numeric,

        errors="coerce"

    )

    X = X.fillna(0)

    model_features = (

        model

        .get_booster()

        .feature_names

    )

    for col in model_features:

        if col not in X.columns:

            X[col] = 0

    X = X[model_features]

    print(

        "Final feature shape:",

        X.shape

    )

    # ======================================================
    # Prediction
    # ======================================================

    probability = model.predict_proba(

        X

    )[:, 1]

    prediction = (

        probability >= 0.5

    ).astype(int)

    # ======================================================
    # Metrics
    # ======================================================

    try:

        auc = roc_auc_score(

            y,

            probability

        )

    except:

        auc = None

    metrics = {


        "dataset":

            "Edge-IIoTset",


        "samples":

            len(df),


        "class_distribution":

            y.value_counts()

            .to_dict(),


        "accuracy":

            accuracy_score(

                y,

                prediction

            ),


        "precision":

            precision_score(

                y,

                prediction,

                zero_division=0

            ),


        "recall":

            recall_score(

                y,

                prediction,

                zero_division=0

            ),


        "f1_score":

            f1_score(

                y,

                prediction,

                zero_division=0

            ),


        "macro_f1":

            f1_score(

                y,

                prediction,

                average="macro",

                zero_division=0

            ),


        "mcc":

            matthews_corrcoef(

                y,

                prediction

            ),


        "roc_auc":

            auc,


        "confusion_matrix":

            confusion_matrix(

                y,

                prediction,

                labels=[0, 1]

            ).tolist(),


        "classification_report":

            classification_report(

                y,

                prediction,

                zero_division=0

            )

    }

    print()

    print(metrics)

    # ======================================================
    # Save
    # ======================================================

    output = (

        PROJECT_ROOT
        /
        "results"
        /
        "edge_validation"

    )

    output.mkdir(

        parents=True,

        exist_ok=True

    )

    with open(

        output /
        "edge_metrics.json",

        "w"

    ) as f:

        json.dump(

            metrics,

            f,

            indent=4

        )

    pd.DataFrame(

        {

            "probability":

                probability,


            "prediction":

                prediction,


            "actual":

                y.values

        }

    ).to_csv(

        output /
        "edge_predictions.csv",

        index=False

    )

    print()

    print(

        "Saved:",

        output

    )

    print()

    print(
        "Edge-IIoTset Validation Completed Successfully"
    )


if __name__ == "__main__":

    main()
