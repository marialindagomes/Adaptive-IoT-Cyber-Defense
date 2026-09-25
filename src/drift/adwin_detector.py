"""
===========================================================
ADWIN Drift Detector

Purpose:
Detect concept drift from streaming prediction errors

Library:
river

===========================================================
"""


from river.drift import ADWIN


# ==========================================================
# CREATE ADWIN
# ==========================================================

def create_adwin():

    detector = ADWIN()

    return detector


# ==========================================================
# DETECT DRIFT
# ==========================================================

def detect_drift(

        error_stream

):

    adwin = create_adwin()

    drift_points = []

    for index, error in enumerate(error_stream):

        adwin.update(

            error

        )

        if adwin.drift_detected:

            drift_points.append(

                index

            )

    return drift_points
