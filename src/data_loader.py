from pathlib import Path
import pandas as pd

DATA_PATH = Path("data") / "dashboard_sample.parquet"
CLIENT_ID_COL = "SK_ID_CURR"

def load_clients_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {DATA_PATH}")
    return pd.read_parquet(DATA_PATH)

def get_client_row(client_id: int) -> pd.Series:
    df = load_clients_data()
    row = df[df[CLIENT_ID_COL] == client_id]
    if row.empty:
        raise ValueError(f"Client {client_id} introuvable.")
    return row.iloc[0]

def get_client_id_col() -> str:
    return CLIENT_ID_COL
