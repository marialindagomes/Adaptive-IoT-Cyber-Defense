"""
===========================================================
Explanation Reliability Score (ERS)

Purpose:
Measure SHAP explanation reliability

===========================================================
"""


import numpy as np


def calculate_shap_stability(

        shap_values

):

    variance = np.var(

        shap_values

    )

    stability = 1 / (1 + variance)

    return stability


def calculate_feature_consistency(

        shap_values

):

    mean_importance = np.mean(

        np.abs(shap_values)

    )

    consistency = mean_importance / (

        mean_importance + 1

    )

    return consistency


def calculate_model_confidence(

        probabilities

):

    confidence = np.mean(

        np.maximum(

            probabilities,

            1 - probabilities

        )

    )

    return confidence


def calculate_ers(

        shap_values,

        probabilities

):

    shap_stability = calculate_shap_stability(

        shap_values

    )

    feature_consistency = calculate_feature_consistency(

        shap_values

    )

    model_confidence = calculate_model_confidence(

        probabilities

    )

    ers = (

        0.5 * shap_stability

        +

        0.3 * feature_consistency

        +

        0.2 * model_confidence

    )

    return {

        "shap_stability":

            float(shap_stability),


        "feature_consistency":

            float(feature_consistency),


        "model_confidence":

            float(model_confidence),


        "ERS":

            float(ers)

    }
