from pathlib import Path
import pandas as pd

P7_DASHBOARD_DATA = Path(
    "/home/nicolasd/code/nicolasdestrac/OpenClassrooms/Projet_7/data/raw/dashboard_sample.parquet"
)
CLIENT_ID_COL = "SK_ID_CURR"

def load_clients_data() -> pd.DataFrame:
    if not P7_DASHBOARD_DATA.exists():
        raise FileNotFoundError(f"Fichier introuvable : {P7_DASHBOARD_DATA}")
    return pd.read_parquet(P7_DASHBOARD_DATA)

def get_client_row(client_id: int) -> pd.Series:
    df = load_clients_data()
    row = df[df[CLIENT_ID_COL] == client_id]
    if row.empty:
        raise ValueError(f"Client {client_id} introuvable.")
    return row.iloc[0]

def get_client_id_col() -> str:
    return CLIENT_ID_COL
