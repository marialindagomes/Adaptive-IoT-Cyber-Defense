"""
===========================================================
Isolation Forest Anomaly Detection Module

Purpose:
Detect unknown / unseen cyber attack behavior

Training:
Only Benign Traffic

Output:
- Anomaly prediction
- Anomaly score

===========================================================
"""


from pathlib import Path

import joblib

import pandas as pd

from sklearn.ensemble import IsolationForest


# ==========================================================
# CREATE MODEL
# ==========================================================

def create_isolation_forest(

        contamination=0.01

):

    model = IsolationForest(

        n_estimators=300,

        contamination=contamination,

        random_state=42,

        n_jobs=-1,

        max_samples="auto"

    )

    return model


# ==========================================================
# TRAIN MODEL
# ==========================================================

def train_isolation_forest(

        X_train,

        model_path="models/anomaly/isolation_forest.pkl"

):

    print("="*60)

    print(
        "Training Isolation Forest"
    )

    print("="*60)

    model = create_isolation_forest()

    model.fit(

        X_train

    )

    model_path = Path(

        model_path

    )

    model_path.parent.mkdir(

        parents=True,

        exist_ok=True

    )

    joblib.dump(

        model,

        model_path

    )

    print(

        "Isolation Forest saved:",

        model_path

    )

    return model


# ==========================================================
# PREDICT ANOMALY
# ==========================================================

def predict_anomaly(

        model,

        X

):

    prediction = model.predict(

        X

    )

    score = model.decision_function(

        X

    )

    result = pd.DataFrame(

        {

            "anomaly_prediction":

                prediction,


            "anomaly_score":

                score

        }

    )

    return result


# ==========================================================
# LOAD MODEL
# ==========================================================

def load_isolation_forest(

        path="models/anomaly/isolation_forest.pkl"

):

    return joblib.load(

        path

    )
