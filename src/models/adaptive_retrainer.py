"""
===========================================================
Adaptive Retrainer Module

Purpose:
Retrain XGBoost model after drift detection

===========================================================
"""


from pathlib import Path

import xgboost as xgb

import json


# ==========================================================
# LOAD TRAINING DATA WINDOW
# ==========================================================

def prepare_retraining_data(

        df

):

    X = df.drop(

        columns=["target"]

    )

    y = df["target"]

    return X, y


# ==========================================================
# RETRAIN MODEL
# ==========================================================

def retrain_xgboost(

        X,

        y,

        old_model=None,

        output_path=None

):

    print(
        "Starting adaptive retraining..."
    )

    model = xgb.XGBClassifier(

        n_estimators=300,

        max_depth=8,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        tree_method="hist",

        eval_metric="logloss",

        random_state=42

    )

    model.fit(

        X,

        y

    )

    if output_path:

        model.save_model(

            output_path

        )

        print(

            "Adaptive model saved:",

            output_path

        )

    return model
