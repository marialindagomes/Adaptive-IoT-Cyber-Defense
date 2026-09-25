"""
===========================================================
Flow Stream Loader

Purpose:
Prepare network traffic flow stream
from CICIoT2023 CSV flow records

Used for:
- Streaming Simulation
- ADWIN Drift Detection
- Adaptive Retraining

===========================================================
"""


from pathlib import Path

import pandas as pd

import numpy as np


# ==========================================================
# LOAD STREAM DATA
# ==========================================================

def load_stream_data(

        path

):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(

            f"Stream dataset not found: {path}"

        )

    df = pd.read_csv(

        path

    )

    return df


# ==========================================================
# CREATE STREAM BATCHES
# ==========================================================

def create_stream_batches(

        df,

        batch_size=1000,

        shuffle=True,

        random_seed=42

):

    data = df.copy()

    if shuffle:

        data = data.sample(

            frac=1,

            random_state=random_seed

        ).reset_index(

            drop=True

        )

    batches = []

    total = len(data)

    for start in range(

        0,

        total,

        batch_size

    ):

        end = start + batch_size

        batch = data.iloc[

            start:end

        ]

        batches.append(

            batch

        )

    return batches


# ==========================================================
# STREAM GENERATOR
# ==========================================================

def stream_generator(

        df,

        batch_size=1000

):

    batches = create_stream_batches(

        df,

        batch_size

    )

    for batch in batches:

        yield batch
