"""
===========================================================
SHAP Explainer Module

Purpose:
Explain XGBoost attack predictions

===========================================================
"""


import shap

import pandas as pd


def calculate_shap_values(

        model,

        X

):

    explainer = shap.TreeExplainer(

        model

    )

    shap_values = explainer.shap_values(

        X

    )

    return shap_values, explainer


def feature_importance(

        shap_values,

        feature_names

):

    importance = pd.DataFrame(

        {

            "feature":

                feature_names,


            "importance":

                abs(shap_values)

                .mean(axis=0)

        }

    )

    importance = importance.sort_values(

        by="importance",

        ascending=False

    )

    return importance
