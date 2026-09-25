"""
===========================================================
Prepare Edge-IIoTset Validation Dataset

Purpose:
Make Edge-IIoTset compatible with CICIoT2023 model

===========================================================
"""


from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():

    print("="*70)

    print(
        "Preparing Edge-IIoTset Validation Dataset"
    )

    print("="*70)

    input_file = (

        PROJECT_ROOT
        /
        "data"
        /
        "processed"
        /
        "edge_iiotset"
        /
        "features.csv"

    )

    output_file = (

        PROJECT_ROOT
        /
        "data"
        /
        "processed"
        /
        "edge_iiotset"
        /
        "validation_ready.csv"

    )

    # CICIoT2023 feature names

    target_features = [

        'flow_duration',
        'Header_Length',
        'Protocol Type',
        'Duration',
        'Rate',
        'Srate',
        'Drate',
        'fin_flag_number',
        'syn_flag_number',
        'rst_flag_number',
        'psh_flag_number',
        'ack_flag_number',
        'ece_flag_number',
        'cwr_flag_number',
        'ack_count',
        'syn_count',
        'fin_count',
        'urg_count',
        'rst_count',
        'HTTP',
        'HTTPS',
        'DNS',
        'Telnet',
        'SMTP',
        'SSH',
        'IRC',
        'TCP',
        'UDP',
        'DHCP',
        'ARP',
        'ICMP',
        'IPv',
        'LLC',
        'Tot sum',
        'Min',
        'Max',
        'AVG',
        'Std',
        'Tot size',
        'IAT',
        'Number',
        'Magnitue',
        'Radius',
        'Covariance',
        'Variance',
        'Weight'
    ]

    chunks = []

    print(
        "Reading Edge-IIoTset..."
    )

    for chunk in pd.read_csv(

        input_file,

        chunksize=20000,

        engine="python",

        on_bad_lines="skip"

    ):

        if "target" not in chunk.columns:

            continue

        chunk["target"] = pd.to_numeric(

            chunk["target"],

            errors="coerce"

        )

        chunk = chunk.dropna(

            subset=["target"]

        )

        chunk["target"] = chunk["target"].astype(int)

        # create missing CIC features

        for col in target_features:

            if col not in chunk.columns:

                chunk[col] = 0

        # keep only required features

        chunk = chunk[

            target_features + ["target"]

        ]

        # numeric conversion

        for col in target_features:

            chunk[col] = pd.to_numeric(

                chunk[col],

                errors="coerce"

            )

        chunk = chunk.fillna(0)

        chunks.append(

            chunk

        )

        print(

            "Processed rows:",

            sum(len(x) for x in chunks)

        )

        # enough validation samples

        if sum(len(x) for x in chunks) >= 100000:

            break

    df = pd.concat(

        chunks,

        ignore_index=True

    )

    df = df.sample(

        frac=1,

        random_state=42

    )

    df.to_csv(

        output_file,

        index=False

    )

    print()

    print(

        "Final shape:",

        df.shape

    )

    print(

        "Saved:",

        output_file

    )

    print(
        "Edge Validation Preparation Completed"
    )


if __name__ == "__main__":

    main()
