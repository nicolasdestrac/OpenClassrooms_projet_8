import numpy as np
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

def plot_bivariate(df, feature_x, feature_y, color=None, client_row=None):
    """
    Nuage de points bi-varié :
    - clipping au 1er–99e percentile
    - coloration optionnelle
    - croix rouge au-dessus pour le client sélectionné
    """

    # --- 1) Sélection des colonnes (sans doublons) ---
    cols = [feature_x, feature_y]
    if color is not None:
        cols.append(color)

    cols = list(dict.fromkeys(cols))   # supprime les doublons proprement

    df_plot = df[cols].dropna()

    # --- 2) Clipping percentile ---
    x1, x99 = np.nanpercentile(df_plot[feature_x].astype(float), [1, 99])
    y1, y99 = np.nanpercentile(df_plot[feature_y].astype(float), [1, 99])

    df_plot = df_plot[
        df_plot[feature_x].astype(float).between(x1, x99)
        & df_plot[feature_y].astype(float).between(y1, y99)
    ]

    # --- 3) Nuage de points ---
    fig = px.scatter(
        df_plot,
        x=feature_x,
        y=feature_y,
        color=color if color else None,
        opacity=0.6,
        height=500,
    )

    # --- 4) Ajout du client ---
    if client_row is not None:
        cx = float(client_row[feature_x])
        cy = float(client_row[feature_y])

        fig.add_scatter(
            x=[cx],
            y=[cy],
            mode="markers",
            name="Client sélectionné",
            marker=dict(
                size=18,
                color="red",
                symbol="x",
                line=dict(width=3, color="black"),
            ),
            showlegend=True,
        )

    fig.update_layout(margin=dict(l=0, r=0, t=10, b=40))
    return fig

def make_risk_gauge(proba: float, threshold: float):
    """
    Jauge linéaire 0-100 % :

    - Fond : dégradé vert -> rouge foncé sur toute la barre 0 → 100
    - Par-dessus : partie proba → 100 recouverte en gris
    - Ligne noire verticale au seuil
    """
    if proba is None:
        proba = 0.0
    proba = float(proba)
    proba = max(0.0, min(1.0, proba))
    value_pct = proba * 100
    threshold_pct = threshold * 100

    N = 200  # résolution de la barre

    fig = go.Figure()

    # --- 1) Dégradé complet 0 -> 100 (toujours le même) ---
    x_full = np.linspace(0, 100, N)
    z_full = np.linspace(0, 1, N)  # 0=vert, 1=rouge foncé

    fig.add_trace(
        go.Heatmap(
            x=x_full,
            y=[0],
            z=[z_full],  # shape (1, N)
            colorscale=[
                [0.0, "#2ecc71"],  # vert
                [0.30, "#f3f044"],  # vert
                [0.5, "#e22828"],  # rouge foncé
                [1.0, "#920000"],  # vert
            ],
            showscale=False,
            hoverinfo="skip",
        )
    )

    # --- 2) Recouvrement gris de proba -> 100 ---
    if value_pct < 100:
        x_grey = np.linspace(value_pct, 100, N)
        z_grey = np.zeros(N)

        fig.add_trace(
            go.Heatmap(
                x=x_grey,
                y=[0],
                z=[z_grey],
                colorscale=[
                    [0.0, "#e0e0e0"],
                    [1.0, "#e0e0e0"],
                ],
                showscale=False,
                hoverinfo="skip",
            )
        )

    # --- 3) Ligne verticale à la proba du client ---
    fig.add_shape(
        type="line",
        x0=threshold_pct,
        x1=threshold_pct,
        y0=-0.3,
        y1=0.3,
        line=dict(color="black", width=2),
    )

    fig.update_xaxes(
        range=[0, 100],
        ticks="outside",
        showgrid=False,
    )
    fig.update_yaxes(
        visible=False,
        range=[-0.25, 0.25],
    )

    fig.update_layout(
        height=30,
        margin=dict(l=0, r=20, t=10, b=10),
    )

    return fig
