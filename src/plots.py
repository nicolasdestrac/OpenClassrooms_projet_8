import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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

def make_risk_gauge(proba: float):
    """
    Jauge de risque : 0% (vert) -> 100% (rouge).
    proba est entre 0 et 1.
    """
    # clamp au cas où
    if proba is None:
        proba = 0.0
    proba = max(0.0, min(1.0, float(proba)))
    value_pct = proba * 100

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value_pct,
            number={"suffix": " %"},
            gauge={
                "axis": {"range": [0, 100]},
                # On laisse la "barre" transparente et on trace un seuil pour la valeur
                "bar": {"color": "rgba(0,0,0,0)"},
                # On approxime un dégradé vert -> rouge avec plusieurs steps
                "steps": [
                    {"range": [0, 10],  "color": "#2ecc71"},  # vert
                    {"range": [10, 30], "color": "#a3d977"},  # vert-jaune
                    {"range": [30, 50], "color": "#f1c40f"},  # jaune
                    {"range": [50, 70], "color": "#e64922"},  # orange
                    {"range": [70, 85], "color": "#d42929"},  # orange-rouge
                    {"range": [85, 100], "color": "#aa0000"}, # rouge écarlate
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.8,
                    "value": value_pct,
                },
            },
        )
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=220,
    )

    return fig
