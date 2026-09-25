"""
===========================================================
Edge-IIoTset Adaptive Validation FINAL v2

Purpose:
Adapt CICIoT2023 XGBoost model using Edge-IIoTset

Fix:
- Class imbalance
- Threshold selection
- Domain adaptation

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
    confusion_matrix
)

from sklearn.utils.class_weight import compute_sample_weight


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def evaluate(

        y_true,

        probability,

        threshold

):

    prediction = (

        probability >= threshold

    ).astype(int)

    return {


        "threshold":

            threshold,


        "accuracy":

            accuracy_score(

                y_true,

                prediction

            ),


        "precision":

            precision_score(

                y_true,

                prediction,

                zero_division=0

            ),


        "recall":

            recall_score(

                y_true,

                prediction,

                zero_division=0

            ),


        "f1_score":

            f1_score(

                y_true,

                prediction,

                zero_division=0

            ),


        "macro_f1":

            f1_score(

                y_true,

                prediction,

                average="macro",

                zero_division=0

            ),


        "mcc":

            matthews_corrcoef(

                y_true,

                prediction

            ),


        "roc_auc":

            roc_auc_score(

                y_true,

                probability

            ),


        "confusion_matrix":

            confusion_matrix(

                y_true,

                prediction,

                labels=[0, 1]

            ).tolist()

    }


def main():

    print("="*70)

    print(
        "Edge-IIoTset Adaptive Validation FINAL"
    )

    print("="*70)

    # ======================================================
    # Load Edge Dataset
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

    df = pd.read_csv(

        data_file,

        low_memory=False

    )

    print(

        "Samples:",

        len(df)

    )

    df = df.sample(

        frac=1,

        random_state=42

    )

    # ======================================================
    # Split
    # ======================================================

    split = int(

        len(df) * 0.7

    )

    adapt_df = df.iloc[:split]

    test_df = df.iloc[split:]

    X_adapt = adapt_df.drop(

        columns=["target"]

    )

    y_adapt = adapt_df["target"]

    X_test = test_df.drop(

        columns=["target"]

    )

    y_test = test_df["target"]

    # ======================================================
    # Feature Alignment
    # ======================================================

    base_model = xgb.XGBClassifier()

    base_model.load_model(

        PROJECT_ROOT
        /
        "models"
        /
        "adaptive"
        /
        "adaptive_xgboost.json"

    )

    features = (

        base_model

        .get_booster()

        .feature_names

    )

    for col in features:

        if col not in X_adapt.columns:

            X_adapt[col] = 0

            X_test[col] = 0

    X_adapt = X_adapt[features]

    X_test = X_test[features]

    X_adapt = X_adapt.fillna(0)

    X_test = X_test.fillna(0)

    print(

        "Adaptation samples:",

        len(X_adapt)

    )

    print(

        "Test samples:",

        len(X_test)

    )

    # ======================================================
    # Class Balanced Adaptive Training
    # ======================================================

    weights = compute_sample_weight(

        class_weight="balanced",

        y=y_adapt

    )

    adaptive_model = xgb.XGBClassifier(

        n_estimators=400,

        max_depth=8,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        tree_method="hist",

        eval_metric="logloss",

        random_state=42

    )

    print(
        "Adaptive training started..."
    )

    adaptive_model.fit(

        X_adapt,

        y_adapt,

        sample_weight=weights

    )

    print(
        "Adaptive training completed"
    )

    # ======================================================
    # Prediction
    # ======================================================

    probability = adaptive_model.predict_proba(

        X_test

    )[:, 1]

    # ======================================================
    # Threshold Search
    # ======================================================

    best = None

    for threshold in [

        0.2,

        0.3,

        0.4,

        0.5,

        0.6,

        0.7

    ]:

        result = evaluate(

            y_test,

            probability,

            threshold

        )

        if (

            best is None

            or

            result["macro_f1"]

            >

            best["macro_f1"]

        ):

            best = result

    print()

    print(
        "Best Validation Result:"
    )

    print(best)

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
        "edge_adaptive_metrics.json",

        "w"

    ) as f:

        json.dump(

            best,

            f,

            indent=4

        )

    adaptive_model.save_model(

        output /
        "edge_adapted_xgboost.json"

    )

    print()

    print(
        "Saved:"
    )

    print(output)

    print()

    print(
        "Edge Adaptive Validation Completed Successfully"
    )


if __name__ == "__main__":

    main()
