"""
===========================================================
Model Evaluation Module

Metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Macro F1
- MCC
- ROC-AUC
- AUPRC
- Confusion Matrix

Used for:
- Baseline Models
- XGBoost
- Future Models

===========================================================
"""


import json
from pathlib import Path
import time


import numpy as np


from sklearn.metrics import (  # type: ignore[reportMissingModuleSource]

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    matthews_corrcoef,

    roc_auc_score,

    average_precision_score,

    confusion_matrix,

    classification_report

)


# ==========================================================
# EVALUATE MODEL
# ==========================================================

def evaluate_model(

        model,

        X_test,

        y_test,

        model_name,

        save_path=None

):

    print("="*60)

    print(
        f"Evaluating: {model_name}"
    )

    print("="*60)

    # Prediction time

    start_time = time.time()

    y_pred = model.predict(

        X_test

    )

    inference_time = (

        time.time()

        -

        start_time

    )

    # Probability prediction

    y_prob = None

    if hasattr(

        model,

        "predict_proba"

    ):

        y_prob = model.predict_proba(

            X_test

        )[:, 1]

    # ======================================================
    # Metrics
    # ======================================================

    results = {


        "model":

            model_name,


        "accuracy":

            float(

                accuracy_score(

                    y_test,

                    y_pred

                )

            ),



        "precision":

            float(

                precision_score(

                    y_test,

                    y_pred,

                    zero_division=0

                )

            ),



        "recall":

            float(

                recall_score(

                    y_test,

                    y_pred,

                    zero_division=0

                )

            ),



        "f1_score":

            float(

                f1_score(

                    y_test,

                    y_pred,

                    zero_division=0

                )

            ),



        "macro_f1":

            float(

                f1_score(

                    y_test,

                    y_pred,

                    average="macro",

                    zero_division=0

                )

            ),



        "mcc":

            float(

                matthews_corrcoef(

                    y_test,

                    y_pred

                )

            ),



        "inference_time_seconds":

            float(

                inference_time

            )

    }

    # ROC-AUC and AUPRC

    if y_prob is not None:

        results["roc_auc"] = float(

            roc_auc_score(

                y_test,

                y_prob

            )

        )

        results["auprc"] = float(

            average_precision_score(

                y_test,

                y_prob

            )

        )

    else:

        results["roc_auc"] = None

        results["auprc"] = None

    # Confusion Matrix

    cm = confusion_matrix(

        y_test,

        y_pred

    )

    results["confusion_matrix"] = (

        cm.tolist()

    )

    # Classification Report

    results["classification_report"] = (

        classification_report(

            y_test,

            y_pred,

            zero_division=0

        )

    )

    # ======================================================
    # SAVE RESULTS
    # ======================================================

    if save_path:

        save_path = Path(

            save_path

        )

        save_path.parent.mkdir(

            parents=True,

            exist_ok=True

        )

        with open(

            save_path,

            "w"

        ) as file:

            json.dump(

                results,

                file,

                indent=4

            )

        print(

            "Results saved:",

            save_path

        )

    return results
