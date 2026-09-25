from pathlib import Path
import pandas as pd


DATA_PATH = Path(
    "data/raw/edge_iiotset/csv"
)


csv_files = sorted(
    DATA_PATH.glob("*.csv")
)


print("="*60)

print(
    "Checking Edge-IIoTset Columns"
)

print("="*60)


for file in csv_files[:5]:

    print("\nFile:")
    print(file.name)

    df = pd.read_csv(
        file,
        nrows=5
    )

    print("\nColumns:")

    for col in df.columns:

        print(col)

    print("-"*60)
