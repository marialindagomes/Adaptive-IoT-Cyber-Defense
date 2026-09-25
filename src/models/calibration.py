"""
===========================================================
Probability Calibration Module

Purpose:
- Calibrate XGBoost probability output
- Improve confidence reliability

Methods:
1. Platt Scaling (sigmoid)
2. Isotonic Regression

Compatible with:
scikit-learn >= 1.8

===========================================================
"""


from pathlib import Path

import joblib

from sklearn.calibration import CalibratedClassifierCV


# ==========================================================
# PLATT SCALING
# ==========================================================

def calibrate_platt(

        model,

        X_calibration,

        y_calibration

):

    print(
        "Applying Platt Scaling..."
    )

    calibrated_model = CalibratedClassifierCV(

        estimator=model,

        method="sigmoid",

        cv=2

    )

    calibrated_model.fit(

        X_calibration,

        y_calibration

    )

    return calibrated_model


# ==========================================================
# ISOTONIC REGRESSION
# ==========================================================

def calibrate_isotonic(

        model,

        X_calibration,

        y_calibration

):

    print(
        "Applying Isotonic Calibration..."
    )

    calibrated_model = CalibratedClassifierCV(

        estimator=model,

        method="isotonic",

        cv=2

    )

    calibrated_model.fit(

        X_calibration,

        y_calibration

    )

    return calibrated_model


# ==========================================================
# SAVE CALIBRATED MODEL
# ==========================================================

def save_calibrated_model(

        model,

        path

):

    path = Path(path)

    path.parent.mkdir(

        parents=True,

        exist_ok=True

    )

    joblib.dump(

        model,

        path

    )

    print(

        "Calibrated model saved:",

        path

    )


# ==========================================================
# LOAD CALIBRATED MODEL
# ==========================================================

def load_calibrated_model(

        path

):

    return joblib.load(

        path

    )
