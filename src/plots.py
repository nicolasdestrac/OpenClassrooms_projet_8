import pandas as pd
import plotly.express as px

def plot_dummy(df: pd.DataFrame):
    """Exemple de fonction de plot Plotly."""
    fig = px.histogram(df, x=df.columns[0])
    return fig
