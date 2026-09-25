"""
===========================================================
Unknown Attack Detection Module

Combines:
1. Calibrated XGBoost probability
2. Isolation Forest anomaly score

Output:
- Normal
- Known Attack
- Unknown Attack

===========================================================
"""


import pandas as pd


# ==========================================================
# DECISION FUNCTION
# ==========================================================

def detect_unknown_attack(

        xgb_probability,

        anomaly_prediction,

        anomaly_score,

        attack_threshold=0.80,

        anomaly_score_threshold=0

):

    decisions = []

    for prob, anomaly, score in zip(

        xgb_probability,

        anomaly_prediction,

        anomaly_score

    ):

        # Unknown attack:
        # Low XGBoost confidence
        # But Isolation Forest sees anomaly

        if (

            anomaly == -1

            and

            prob < attack_threshold

        ):

            decision = "Unknown Attack"

        # Known attack

        elif prob >= attack_threshold:

            decision = "Known Attack"

        # Normal

        else:

            decision = "Normal"

        decisions.append(

            decision

        )

    return decisions


# ==========================================================
# CREATE REPORT
# ==========================================================

def create_detection_report(

        y_true,

        xgb_probability,

        anomaly_prediction,

        anomaly_score

):

    report = pd.DataFrame(

        {

            "true_label":

                y_true,


            "xgb_probability":

                xgb_probability,


            "anomaly_prediction":

                anomaly_prediction,


            "anomaly_score":

                anomaly_score

        }

    )

    report["final_detection"] = detect_unknown_attack(

        xgb_probability,

        anomaly_prediction,

        anomaly_score

    )

    return report
