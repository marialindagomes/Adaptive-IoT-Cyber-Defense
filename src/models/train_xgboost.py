"""
===========================================================
XGBoost Training Module

Main Cyber Threat Detection Model

Features:
- Class imbalance handling
- Early stopping
- Model saving
- Feature importance extraction

===========================================================
"""


from pathlib import Path
from importlib import import_module

import pandas as pd


# ==========================================================
# CREATE MODEL
# ==========================================================

def create_xgboost_model(scale_pos_weight=1):

    try:
        xgb = import_module("xgboost")
    except ImportError as exc:
        raise ImportError(
            "XGBoost is required to train this model. "
            "Install it with: pip install xgboost"
        ) from exc

    model = xgb.XGBClassifier(

        n_estimators=1000,

        learning_rate=0.05,

        max_depth=8,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="binary:logistic",

        eval_metric="logloss",

        tree_method="hist",

        random_state=42,

        scale_pos_weight=scale_pos_weight,

        n_jobs=-1

    )

    return model


# ==========================================================
# TRAIN MODEL
# ==========================================================

def train_xgboost(

        X_train,

        y_train,

        X_val,

        y_val,

        model_path="models/xgboost/xgboost_model.json"

):

    print("="*60)

    print(
        "Training XGBoost Model"
    )

    print("="*60)

    # Calculate class imbalance ratio

    negative = sum(
        y_train == 0
    )

    positive = sum(
        y_train == 1
    )

    scale_pos_weight = (

        negative /

        positive

    )

    print(

        "Negative samples:",

        negative

    )

    print(

        "Positive samples:",

        positive

    )

    print(

        "Scale Pos Weight:",

        scale_pos_weight

    )

    model = create_xgboost_model(

        scale_pos_weight

    )

    # Train

    model.fit(

        X_train,

        y_train,

        eval_set=[

            (

                X_val,

                y_val

            )

        ],

        verbose=100

    )

    # Save model

    model_path = Path(

        model_path

    )

    model_path.parent.mkdir(

        parents=True,

        exist_ok=True

    )

    model.save_model(

        model_path

    )

    print(

        "\nModel saved:",

        model_path

    )

    return model


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

def save_feature_importance(

        model,

        feature_names,

        output_path="results/xgboost/feature_importance.csv"

):

    importance = pd.DataFrame(

        {

            "feature":

                feature_names,


            "importance":

                model.feature_importances_

        }

    )

    importance = importance.sort_values(

        by="importance",

        ascending=False

    )

    output_path = Path(

        output_path

    )

    output_path.parent.mkdir(

        parents=True,

        exist_ok=True

    )

    importance.to_csv(

        output_path,

        index=False

    )

    print(

        "Feature importance saved:",

        output_path

    )
