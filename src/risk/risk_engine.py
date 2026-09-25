"""
===========================================================
Threat Risk Engine v2

Purpose:
Calculate balanced cyber threat risk score

Inputs:
- attack probability
- anomaly score
- ERS

Output:
- risk score
- threat level

===========================================================
"""


import numpy as np


# ==========================================================
# Normalize anomaly score
# ==========================================================

def normalize_anomaly(

        anomaly_score

):

    # Isolation Forest:
    # higher = normal
    # lower = anomaly

    anomaly = 1 - anomaly_score

    anomaly = np.clip(

        anomaly,

        0,

        1

    )

    return anomaly


# ==========================================================
# Risk Score
# ==========================================================

def calculate_risk_score(

        attack_probability,

        anomaly_score,

        ers_score

):

    anomaly = normalize_anomaly(

        anomaly_score

    )

    risk = (

        0.45 * attack_probability

        +

        0.35 * anomaly

        +

        0.20 * ers_score

    )

    risk = np.clip(

        risk,

        0,

        1

    )

    return float(risk)


# ==========================================================
# Risk Classification
# ==========================================================

def classify_risk(

        risk_score

):

    if risk_score < 0.4:

        return "Low"

    elif risk_score < 0.7:

        return "Medium"

    else:

        return "High"
