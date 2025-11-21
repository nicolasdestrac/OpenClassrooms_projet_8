import pandas as pd
import plotly.express as px

def plot_client_vs_population(df: pd.DataFrame, client_row: pd.Series, feature: str):
    """
    Histogramme + boxplot de la feature sur la population,
    avec une ligne verticale pour la valeur du client.
    """
    fig = px.histogram(
        df,
        x=feature,
        nbins=30,
        opacity=0.7,
        marginal="box"
    )
    fig.add_vline(
        x=client_row[feature],
        line_width=3,
        line_dash="dash",
    )
    fig.update_layout(
        title=f"Distribution de {feature} et position du client",
        xaxis_title=feature,
        yaxis_title="Nombre de clients"
    )
    return fig

def plot_bivariate(df: pd.DataFrame, feature_x: str, feature_y: str, color: str | None = None):
    """
    Nuage de points pour analyse bi-variée entre deux variables.
    """
    fig = px.scatter(
        df,
        x=feature_x,
        y=feature_y,
        color=color
    )
    fig.update_layout(
        title=f"Analyse bi-variée : {feature_x} vs {feature_y}",
        xaxis_title=feature_x,
        yaxis_title=feature_y
    )
    return fig
