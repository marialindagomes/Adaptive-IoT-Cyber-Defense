"""
===========================================================
Final IEEE Figure 3
ADWIN Drift Detection + Adaptive Retraining

Uses real streaming result files only.

===========================================================
"""


from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent


OUTPUT = ROOT / "paper_figures_final"

OUTPUT.mkdir(
    exist_ok=True
)


plt.rcParams.update({

    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12

})


def main():

    # -----------------------------
    # Load drift report
    # -----------------------------

    drift_file = (

        ROOT
        /
        "results"
        /
        "drift"
        /
        "final_supervised_drift_report.json"

    )

    with open(drift_file) as f:

        drift = json.load(f)

    drift_point = drift["drift_points"][0]

    total_samples = drift["total_samples"]

    # -----------------------------
    # Load stream summary
    # -----------------------------

    stream_file = (

        ROOT
        /
        "results"
        /
        "streaming"
        /
        "stream_summary.csv"

    )

    if not stream_file.exists():

        raise FileNotFoundError(

            "stream_summary.csv not found"

        )

    df = pd.read_csv(stream_file)

    print(df.head())

    print(df.columns)

    # Detect columns automatically

    sample_col = None

    error_col = None

    for c in df.columns:

        name = c.lower()

        if (

            "sample" in name

            or

            "index" in name

            or

            "time" in name

        ):

            sample_col = c

        if (

            "error" in name

            or

            "loss" in name

            or

            "rate" in name

        ):

            error_col = c

    if sample_col is None:

        sample_col = df.columns[0]

    if error_col is None:

        error_col = df.columns[1]

    x = df[sample_col]

    y = df[error_col]

    # -----------------------------
    # Plot
    # -----------------------------

    fig, ax = plt.subplots(

        figsize=(8, 4.5)

    )

    ax.plot(

        x,

        y,

        linewidth=1.8,

        label="Prediction Error"

    )

    ax.axvline(

        drift_point,

        linestyle="--",

        linewidth=2,

        label=f"ADWIN Drift Point ({drift_point:,})"

    )

    ax.annotate(

        "Adaptive Retraining\n200,000 samples",

        xy=(

            drift_point,

            y.iloc[-1]

        ),

        xytext=(

            drift_point * 0.65,

            y.max()*0.85

        ),

        arrowprops={

            "arrowstyle": "->",

            "linewidth": 1.5

        },

        fontsize=10

    )

    ax.set_title(

        "ADWIN Concept Drift Detection and Adaptive Retraining"

    )

    ax.set_xlabel(

        "Stream Samples"

    )

    ax.set_ylabel(

        "Prediction Error"

    )

    ax.grid(

        alpha=0.3

    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(

        OUTPUT /

        "Figure3_ADWIN_Adaptive_Final.png",

        dpi=600,

        bbox_inches="tight"

    )

    plt.close()

    print(

        "Figure3_ADWIN_Adaptive_Final.png generated"

    )


if __name__ == "__main__":

    main()
