import pandas as pd
from pathlib import Path

def ingest(raw_data_path: Path) -> pd.DataFrame:

    return pd.read_csv(raw_data_path)

