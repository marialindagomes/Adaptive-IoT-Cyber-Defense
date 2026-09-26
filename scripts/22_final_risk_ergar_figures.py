"""
========================================================
Final IEEE Risk and ERGAR Figures

Figure 4(a): Threat Risk Distribution
Figure 4(b): ERGAR Response Decision

========================================================
"""


from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parent.parent


OUT = ROOT / "paper_figures_final"

OUT.mkdir(
    exist_ok=True
)


plt.rcParams.update({

    "font.family": "serif",
    "font.size": 12

})


# =====================================================
# Figure 4(a)
# Risk Distribution
# =====================================================

def risk_distribution():

    labels = [

        "High Risk",

        "Medium Risk"

    ]

    values = [

        4980,

        20

    ]

    fig, ax = plt.subplots(

        figsize=(5, 5)

    )

    wedges, texts, autotexts = ax.pie(

        values,

        labels=labels,

        autopct="%1.1f%%",

        startangle=90,

        pctdistance=0.75,

        wedgeprops={

            "width": 0.35,

            "edgecolor": "white"

        }

    )

    ax.text(

        0,

        0,

        "Risk\nAssessment",

        ha="center",

        va="center",

        fontsize=14,

        fontweight="bold"

    )

    ax.set_title(

        "Threat Risk Distribution",

        fontsize=15,

        fontweight="bold"

    )

    plt.tight_layout()

    plt.savefig(

        OUT /

        "Figure4a_Risk_Distribution.png",

        dpi=600,

        bbox_inches="tight"

    )

    plt.close()


# =====================================================
# Figure 4(b)
# ERGAR Decision Flow
# =====================================================

def ergar_flow():

    fig, ax = plt.subplots(

        figsize=(8, 5)

    )

    ax.axis("off")

    boxes = [


        (

            "Risk Engine",

            0.4,

            0.75

        ),


        (

            "HIGH RISK\n\nBLOCK_AND_ISOLATE\n4980",

            0.15,

            0.45

        ),


        (

            "MEDIUM RISK\n\nMONITOR_AND_LOG\n20",

            0.65,

            0.45

        )

    ]

    for text, x, y in boxes:

        patch = FancyBboxPatch(

            (x, y),

            0.25,

            0.15,

            boxstyle="round,pad=0.03",

            linewidth=1.5


        )

        ax.add_patch(patch)

        ax.text(

            x+0.125,

            y+0.075,

            text,

            ha="center",

            va="center",

            fontsize=12,
            fontweight="bold",
            color="white"

        )

    # arrows

    arrows = [
        ((0.525, 0.75), (0.275, 0.60)),

        ((0.525, 0.75), (0.775, 0.60))


    ]

    for start, end in arrows:

        ax.add_patch(

            FancyArrowPatch(

                start,

                end,

                arrowstyle="->",

                mutation_scale=15,

                linewidth=1.5

            )

        )

    ax.set_title(

        "ERGAR Automated Response Decision",

        fontsize=15,

        fontweight="bold"

    )

    plt.tight_layout()

    plt.savefig(

        OUT /

        "Figure4b_ERGAR_Response.png",

        dpi=600,

        bbox_inches="tight"

    )

    plt.close()


if __name__ == "__main__":

    risk_distribution()

    ergar_flow()

    print(

        "Risk and ERGAR figures generated successfully"

    )
