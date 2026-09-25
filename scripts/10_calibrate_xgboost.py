"""
===========================================================
XGBoost Probability Calibration Pipeline

Methods:
1. Platt Scaling
2. Isotonic Regression

===========================================================
"""


from pathlib import Path
import json
import importlib.util

import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    accuracy_score
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_module(name, path):

    spec = importlib.util.spec_from_file_location(
        name,
        path
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    return module


# Load modules
label_module = load_module(
    "label_mapper",
    PROJECT_ROOT / "src/data/label_mapper.py"
)


clean_module = load_module(
    "clean_data",
    PROJECT_ROOT / "src/data/clean_data.py"
)


feature_module = load_module(
    "feature_processing",
    PROJECT_ROOT / "src/data/feature_processing.py"
)


cal_module = load_module(
    "calibration",
    PROJECT_ROOT / "src/models/calibration.py"
)


map_labels = label_module.map_ciciot2023_labels

clean_dataset = clean_module.clean_dataset

process_features = feature_module.process_features

calibrate_platt = cal_module.calibrate_platt

calibrate_isotonic = cal_module.calibrate_isotonic

save_calibrated_model = cal_module.save_calibrated_model


def load_validation_data():

    with open(
        PROJECT_ROOT /
        "results/split/validation_files.json"
    ) as f:

        files = json.load(f)

    data_path = (
        PROJECT_ROOT /
        "data/raw/ciciot2023/csv"
    )

    dfs = []

    print("Loading validation files...")

    for file in files:

        print("Reading:", file)

        df = pd.read_csv(

            data_path / file,

            nrows=50000

        )

        dfs.append(df)

    return pd.concat(
        dfs,
        ignore_index=True
    )


def main():

    print("="*60)

    print(
        "XGBoost Probability Calibration"
    )

    print("="*60)

    # Load XGBoost model

    model = xgb.XGBClassifier()

    model.load_model(

        str(

            PROJECT_ROOT /
            "models/xgboost/xgboost_model.json"

        )

    )

    print(
        "XGBoost model loaded"
    )

    # Validation data

    df = load_validation_data()

    df = map_labels(df)

    df = clean_dataset(

        df,

        remove_cols=["label"]

    )

    df = process_features(df)

    X_val = df.drop(
        columns=["target"]
    )

    y_val = df["target"]

    print(
        "Validation shape:",
        X_val.shape
    )

    # Calibration

    platt_model = calibrate_platt(

        model,

        X_val,

        y_val

    )

    isotonic_model = calibrate_isotonic(

        model,

        X_val,

        y_val

    )

    results = {}

    for name, calibrated in [

        ("platt", platt_model),

        ("isotonic", isotonic_model)

    ]:

        prob = calibrated.predict_proba(

            X_val

        )[:, 1]

        pred = (

            prob >= 0.5

        ).astype(int)

        results[name] = {


            "brier_score":

                float(

                    brier_score_loss(

                        y_val,

                        prob

                    )

                ),


            "log_loss":

                float(

                    log_loss(

                        y_val,

                        prob

                    )

                ),


            "accuracy":

                float(

                    accuracy_score(

                        y_val,

                        pred

                    )

                )

        }

    print("\nCalibration Results:")

    print(results)

    best = min(

        results,

        key=lambda x:

        results[x]["brier_score"]

    )

    print(
        "Best calibration:",
        best
    )

    model_to_save = (

        platt_model

        if best == "platt"

        else isotonic_model

    )

    save_calibrated_model(

        model_to_save,

        "models/xgboost/calibrated_xgboost.pkl"

    )

    output = (

        PROJECT_ROOT /
        "results/calibration"

    )

    output.mkdir(

        parents=True,

        exist_ok=True

    )

    with open(

        output /
        "calibration_metrics.json",

        "w"

    ) as f:

        json.dump(

            results,

            f,

            indent=4

        )

    print(
        "Calibration Completed Successfully"
    )


if __name__ == "__main__":

    main()
