from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent


BASELINE_DIR = ROOT / "results" / "baseline"

XGB_FILE = ROOT / "results" / "xgboost" / "xgboost_metrics.json"


OUT_TABLE = ROOT / "paper_assets_final" / "tables"
OUT_FIG = ROOT / "paper_assets_final" / "figures"


OUT_TABLE.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)


rows = []


# Baseline models

for file in BASELINE_DIR.glob("*.json"):

    with open(file) as f:
        data = json.load(f)

    rows.append({

        "Model": data["model"],

        "Accuracy": data["accuracy"],

        "Precision": data["precision"],

        "Recall": data["recall"],

        "F1-score": data["f1_score"],

        "Macro-F1": data["macro_f1"],

        "MCC": data["mcc"],

        "ROC-AUC": data["roc_auc"]

    })


# XGBoost add

with open(XGB_FILE) as f:

    xgb = json.load(f)


rows.append({

    "Model": "XGBoost",

    "Accuracy": xgb["accuracy"],

    "Precision": xgb["precision"],

    "Recall": xgb["recall"],

    "F1-score": xgb["f1_score"],

    "Macro-F1": xgb["macro_f1"],

    "MCC": xgb["mcc"],

    "ROC-AUC": xgb["roc_auc"]

})


df = pd.DataFrame(rows)


# sort order

order = [

    "Logistic Regression",

    "Decision Tree",

    "Random Forest",

    "MLP",

    "XGBoost"

]


df["Model"] = pd.Categorical(

    df["Model"],

    categories=order,

    ordered=True

)


df = df.sort_values("Model")


# Save Table

df.to_csv(

    OUT_TABLE /

    "Table3_Baseline_Comparison.csv",

    index=False

)


print(df)


# Figure - F1 comparison

plt.figure(figsize=(8, 5))


plt.bar(

    df["Model"].astype(str),

    df["F1-score"]

)


plt.ylabel(

    "F1-score"

)


plt.title(

    "Baseline Model Comparison"

)


plt.xticks(

    rotation=35,

    ha="right"

)


plt.tight_layout()


plt.savefig(

    OUT_FIG /

    "Figure2_Baseline_Comparison.png",

    dpi=300

)


plt.close()


print(
    "Baseline table and figure generated successfully"
)
