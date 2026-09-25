"""
===========================================================
ERGAR Response Pipeline

Input:
final_risk_assessment.csv

Output:
response_actions.csv

===========================================================
"""


from pathlib import Path

import pandas as pd

import json

import importlib.util


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_module(name, path):

    spec = importlib.util.spec_from_file_location(

        name,

        path

    )

    module = importlib.util.module_from_spec(

        spec

    )

    spec.loader.exec_module(

        module

    )

    return module


ergar = load_module(

    "ergar",

    PROJECT_ROOT
    /
    "src"
    /
    "response"
    /
    "ergar.py"

)


def main():

    print("="*70)

    print(
        "ERGAR Adaptive Response Engine"
    )

    print("="*70)

    risk_file = (

        PROJECT_ROOT
        /
        "results"
        /
        "risk"
        /
        "final_risk_assessment.csv"

    )

    df = pd.read_csv(

        risk_file

    )

    responses = []

    for _, row in df.iterrows():

        response = ergar.generate_response(

            row["threat_level"]

        )

        responses.append(

            {

                "sample_id":

                row["sample_id"],


                "risk_score":

                row["risk_score"],


                "threat_level":

                row["threat_level"],


                "response_action":

                response["action"],


                "priority":

                response["priority"]

            }

        )

    result = pd.DataFrame(

        responses

    )

    output = (

        PROJECT_ROOT
        /
        "results"
        /
        "response"

    )

    output.mkdir(

        parents=True,

        exist_ok=True

    )

    result.to_csv(

        output /
        "response_actions.csv",

        index=False

    )

    report = {


        "total_samples":

            len(result),


        "actions":

            result["response_action"]

            .value_counts()

            .to_dict()

    }

    with open(

        output /
        "response_report.json",

        "w"

    ) as f:

        json.dump(

            report,

            f,

            indent=4

        )

    print(report)

    print(
        "ERGAR Response Completed Successfully"
    )


if __name__ == "__main__":

    main()
