from pathlib import Path
import pandas as pd


DATA_PATH = Path(
    "data/raw/edge_iiotset/csv"
)


csv_files = sorted(
    DATA_PATH.glob("*.csv")
)


for file in csv_files[:3]:

    print("\nFile:")
    print(file.name)

    df = pd.read_csv(
        file,
        nrows=10
    )

    print(
        df[
            [
                "Attack_label",
                "Attack_type"
            ]
        ]
    )
