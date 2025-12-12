import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src.api_client import (
    get_schema,
    predict,
    explain,
    check_health,
)
from src.data_loader import (
    load_clients_data,
    get_client_row,
    get_client_id_col,
)
from src.plots import (
    plot_client_vs_population,
    plot_bivariate,
    make_risk_gauge,
)
from src.feature_label import (
    feature_label,
)

# ---------------------------
# Config de la page
# ---------------------------
st.set_page_config(
    page_title="Dashboard scoring crédit - Projet 8",
    layout="wide"
)

# ---------------------------
# Thème & accessibilité (WCAG)
# ---------------------------
st.markdown(
    """
    <style>
    /* --------- Base typographique --------- */
    html, body, [class*="css"] {
        color: #111111 !important;
        font-size: 16px;
        line-height: 1.5;
    }

    /* Titres Streamlit (h1, h2, h3) */
    h1, h2, h3, h4 {
        color: #111111 !important;
        font-weight: 700 !important;
    }

    /* Texte "caption" de Streamlit : on le fonce aussi */
    .st-emotion-cache-1v0mbdj,  /* class fréquente pour ally_small */
    .stCaption, .css-145kmo2, .css-12ttj6m {
        color: #333333 !important;
        font-size: 0.95rem !important;
    }

    /* Lien et boutons : forte visibilité + focus clavier */
    a, a:visited {
        color: #0057B8 !important;
        text-decoration: underline;
    }
    a:hover {
        color: #003f82 !important;
    }
    button, .stButton button {
        min-height: 40px;
        padding: 0.4rem 0.9rem;
        border-radius: 4px;
        border: 1px solid #004a99;
        background-color: #0057B8;
        color: #ffffff !important;
        font-weight: 600;
    }
    button:hover, .stButton button:hover {
        background-color: #004a99;
    }

    /* Focus clavier bien visible : WCAG 2.4.7 */
    a:focus-visible,
    button:focus-visible,
    [role="button"]:focus-visible,
    input:focus-visible,
    select:focus-visible,
    textarea:focus-visible {
        outline: 3px solid #ff8800 !important;
        outline-offset: 2px;
    }

    /* Séparateurs visuels légers entre blocs principaux */
    section.main > div {
        border-radius: 0;
    }

    /* Sidebar : texte plus foncé aussi */
    [data-testid="stSidebar"] {
        color: #111111 !important;
    }

    /* Amélioration des radios / cases à cocher pour clic + large */
    label[data-baseweb="radio"] > div,
    label[data-baseweb="checkbox"] > div {
        padding-top: 2px;
        padding-bottom: 2px;
    }

    /* Eviter que les légendes de graphiques soient trop petites */
    .legendtext {
        font-size: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data
def cached_load_clients_data():
    return load_clients_data()

@st.cache_data
def cached_get_schema():
    try:
        return get_schema()
    except Exception:
        return []

def to_json_serializable(value):
    import math

    # Convertit types numpy en types Python
    if isinstance(value, (np.integer, np.int32, np.int64)):
        value = int(value)
    elif isinstance(value, (np.floating, np.float32, np.float64)):
        value = float(value)
    elif isinstance(value, (np.bool_,)):
        value = bool(value)

    # Remplacement des valeurs non JSON-compliant
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None  # ou 0, selon ton modèle
    return value

def build_features_from_row(row: pd.Series, schema: list[str] | None = None) -> dict:
    features = {}

    if schema:
        for col in schema:
            raw_value = row[col] if col in row.index else None
            features[col] = to_json_serializable(raw_value)
    else:
        for col, raw_value in row.to_dict().items():
            features[col] = to_json_serializable(raw_value)

    return features

@st.cache_data(show_spinner=False)
def compute_global_importance(n_samples: int, schema: list[str] | None):
    """
    Importance globale = moyenne des |SHAP| sur un échantillon de clients.
    On appelle l'endpoint /explain pour n_samples clients au max.
    Résultat : DataFrame (feature, mean_abs_shap) trié décroissant.
    """
    from src.api_client import explain  # import local pour éviter boucles
    df = cached_load_clients_data()

    n = min(n_samples, len(df))
    df_sample = df.sample(n, random_state=42)

    agg: dict[str, list[float]] = {}
    for _, row in df_sample.iterrows():
        feats = build_features_from_row(row, schema if schema else None)
        try:
            res = explain(feats)
            contrib = res.get("contrib", {})
        except Exception:
            continue

        for name, v in contrib.items():
            agg.setdefault(name, []).append(abs(float(v)))

    if not agg:
        return pd.DataFrame(columns=["mean_abs_shap"])

    mean_abs = {k: float(np.mean(vals)) for k, vals in agg.items() if len(vals) > 0}
    df_imp = pd.DataFrame.from_dict(mean_abs, orient="index", columns=["mean_abs_shap"])
    df_imp = df_imp.sort_values("mean_abs_shap", ascending=False)
    return df_imp

def a11y_caption(text: str):
    """
    Alternative à ally_small avec un contraste suffisant.
    """
    st.markdown(
        f"<p style='color:#333333; font-size:0.95rem;'>{text}</p>",
        unsafe_allow_html=True,
    )

def a11y_small(text: str):
    """
    Pour un petit texte d'aide (par ex. sous les graphiques).
    """
    st.markdown(
        f"<p style='color:#333333; font-size:0.9rem;'>{text}</p>",
        unsafe_allow_html=True,
    )

# ---------------------------
# Sidebar : configuration
# ---------------------------
st.sidebar.title("Configuration")

# Santé de l'API
with st.sidebar.expander("Statut de l'API"):
    try:
        health = check_health()
        st.success("✅ API disponible")
        st.json(health)
    except Exception as e:
        st.error("❌ Impossible de joindre l'API")
        a11y_small(str(e))

# Chargement des données clients
try:
    df_clients = cached_load_clients_data()
    client_id_col = get_client_id_col()
    client_ids = sorted(df_clients[client_id_col].unique().tolist())
except Exception as e:
    st.error(f"Erreur au chargement des données clients: {e}")
    st.stop()

selected_client_id = st.sidebar.selectbox(
    "Sélectionner un client",
    options=client_ids
)

section = st.sidebar.radio(
    "Section",
    [
        "Vue d'ensemble",
        "Interprétation du score",
        "Comparaison population",
        "Analyse bi-variée",
        "Simulation / Modification client"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Ce dashboard a pour objectif d'aider le chargé de relation client "
    "à expliquer les décisions d'octroi de crédit de manière transparente."
)

# ---------------------------
# Récupération des infos client + prédiction
# ---------------------------

try:
    client_row = get_client_row(selected_client_id)
except Exception as e:
    st.error(f"Erreur lors de la récupération du client: {e}")
    st.stop()

schema = cached_get_schema()
features = build_features_from_row(client_row, schema=schema if schema else None)

# Appel API /predict
try:
    pred_result = predict(features)
    proba = float(pred_result.get("probability", 0.0))
    prediction = int(pred_result.get("prediction", 0))
    threshold = float(pred_result.get("threshold", 0.5))
except Exception as e:
    st.error(f"Erreur lors de l'appel à l'API /predict: {e}")
    proba = None
    prediction = None
    threshold = 0.5

# ---------------------------
# Section : Vue d'ensemble
# ---------------------------
if section == "Vue d'ensemble":
    st.title("Vue d'ensemble du client")

    col1, col2, col3 = st.columns(3)

    # --- Carte 1 : Décision du modèle ---
    with col1:
        st.subheader("Décision du modèle")
        if prediction is not None:
            decision_label = "✅ ACCORDÉ" if prediction == 0 else "❌ REFUSÉ"
            st.metric("Décision modèle", decision_label)
        else:
            st.metric("Décision modèle", "Indisponible")

    # --- Carte 2 : Probabilité & niveau de risque ---
    with col2:
        st.subheader("Niveau de risque")
        if proba is not None:
            proba_pct = proba * 100

            if proba < 0.15:
                risk_level = "Faible"
                risk_expl = "Le risque estimé est faible."
            elif proba < 0.40:
                risk_level = "Modéré"
                risk_expl = "Le risque estimé est modéré."
            else:
                risk_level = "Élevé"
                risk_expl = "Le risque estimé est élevé."

            st.metric("Probabilité de défaut", f"{proba_pct:.1f} %")
            st.write(f"Niveau de risque : **{risk_level}**")
            a11y_small(risk_expl)

            gauge_fig = make_risk_gauge(proba, threshold)
            st.plotly_chart(
                gauge_fig,
                use_container_width=True,
                config={
                    "staticPlot": True,
                    "displayModeBar": False,
                },
            )

            a11y_small(
                f"La jauge ci-dessus représente la probabilité de défaut du client : "
                f"**{proba_pct:.1f} %** sur une échelle de 0 à 100 %. "
                "Plus la jauge se rapproche de la droite, plus le risque estimé est élevé."
            )

        else:
            st.write("Probabilité de défaut indisponible.")


    # --- Carte 3 : Seuil & distance ---
    with col3:
        st.subheader("Seuil de décision")
        if proba is not None:
            distance = proba - threshold
            st.write(f"Seuil de décision : **{threshold:.1%}**")
            st.write(f"Distance au seuil : **{distance:.1%}**")

            if proba >= threshold:
                st.info(
                    "La probabilité de défaut est **au-dessus** du seuil : "
                    "le modèle a tendance à **refuser** ce crédit."
                )
            else:
                st.info(
                    "La probabilité de défaut est **en-dessous** du seuil : "
                    "le modèle a tendance à **accepter** ce crédit."
                )
        else:
            st.write("Seuil de décision : **indisponible**")

    st.markdown("---")
    st.subheader("Résumé à communiquer au client")

    if proba is not None and prediction is not None:
        decision_label = "accepté" if prediction == 0 else "refusé"
        seuil_pct = threshold * 100
        proba_pct = proba * 100

        if proba < 0.10:
            risk_level = "un risque faible"
        elif proba < 0.30:
            risk_level = "un risque modéré"
        else:
            risk_level = "un risque élevé"

        st.markdown(
            f"""
            > Pour ce client, le modèle estime la probabilité de non-remboursement à **{proba_pct:.1f} %**,
            > ce qui correspond à **{risk_level}** par rapport aux autres dossiers.
            > Le seuil de décision est fixé à **{seuil_pct:.0f} %**, le dossier est donc **{decision_label}** par le modèle.
            """
        )
    else:
        a11y_small("Résumé indisponible car la prédiction n'a pas pu être calculée.")

    st.markdown("---")
    st.subheader("Caractéristiques principales du client")

    # On remplace les noms techniques par des labels FR uniquement pour l'affichage
    row_display = client_row.copy()
    row_display.index = [feature_label(idx) for idx in row_display.index]

    st.dataframe(
        pd.DataFrame(row_display).T,
        use_container_width=True
    )

# ---------------------------
# Section : Interprétation du score (SHAP local)
# ---------------------------
elif section == "Interprétation du score":
    st.title("Interprétation du score (importance locale vs globale)")

    # --- Importance locale : SHAP pour ce client ---
    try:
        explain_result = explain(features)
        contrib = explain_result.get("contrib", {})
    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API /explain: {e}")
        contrib = {}

    col_local, col_global = st.columns(2)

    with col_local:
        st.subheader("Importance locale (ce client)")

        if contrib:
            # contrib = {"feature_name": shap_value, ...}
            local_df = (
                pd.DataFrame.from_dict(contrib, orient="index", columns=["shap_value"])
                .assign(abs_shap=lambda d: d["shap_value"].abs())
            )

            # Tri par importance absolue décroissante
            local_df = local_df.sort_values("abs_shap", ascending=False).head(15)

            # Libellés FR
            local_df["feature"] = local_df.index
            local_df["label"] = local_df["feature"].map(feature_label)

            # Direction : vers refus (risque ↑) / vers accord (risque ↓)
            local_df["direction"] = np.where(
                local_df["shap_value"] >= 0,
                "Risque ↑ (pousse vers le refus)",
                "Risque ↓ (pousse vers l'accord)"
            )

            # On remet à plat pour Plotly
            plot_df = local_df.reset_index(drop=True)

            fig_local = px.bar(
                plot_df,
                x="label",
                y="shap_value",
                color="direction",
                color_discrete_map={
                    # Rouge pour “risque ↑”
                    "Risque ↑ (pousse vers le refus)": "#CC0000",
                    # Bleu pour “risque ↓”
                    "Risque ↓ (pousse vers l'accord)": "#0057B8",
                },
            )

            fig_local.update_layout(
                xaxis_title="Variables",
                yaxis_title="Impact sur le risque (valeur SHAP)",
                xaxis_tickangle=-45,
                margin=dict(l=0, r=0, t=0, b=120),
                showlegend=True,
            )
            fig_local.add_hline(y=0, line_width=1, line_color="black")

            st.plotly_chart(fig_local, use_container_width=True, config={"displayModeBar": False})

            # Texte explicatif simple
            a11y_small(
                "En bleu : les variables qui diminuent le risque et poussent le modèle à **accepter**."
                "En rouge : celles qui augmentent le risque et poussent vers le **refus**."
            )

            # Top 3 pour le résumé texte
            top_feats = (
                plot_df.sort_values("abs_shap", ascending=False)
                .head(3)["label"]
                .tolist()
            )
            st.markdown(
                "Pour ce client, les variables qui ont le plus pesé sont : "
                + ", ".join(f"**{name}**" for name in top_feats)
                + "."
            )

        else:
            st.info("Les contributions SHAP ne sont pas disponibles pour ce client.")


    # --- Importance globale : moyenne SHAP sur un échantillon de clients ---
    with col_global:
        st.subheader("Importance globale (population)")

        with st.spinner("Calcul de l'importance globale sur un échantillon de clients..."):
            global_imp = compute_global_importance(
                n_samples=200,
                schema=schema if schema else None,
            )

        if (global_imp is not None) and (not global_imp.empty):
            # global_imp : index = feature, colonne = importance moyenne (abs SHAP)
            # adapte le nom de colonne si besoin : "importance" ou "mean_abs_shap"
            col_name = "importance" if "importance" in global_imp.columns else global_imp.columns[0]

            gdf = global_imp.copy()
            gdf["feature"] = gdf.index
            gdf["label"] = gdf["feature"].map(feature_label)

            gdf = gdf.sort_values(col_name, ascending=False).head(15)
            plot_gdf = gdf.reset_index(drop=True)

            fig_global = px.bar(
                plot_gdf,
                x="label",
                y=col_name,
            )
            fig_global.update_layout(
                xaxis_title="Variables",
                yaxis_title="Importance moyenne (|SHAP|)",
                xaxis_tickangle=-45,
                margin=dict(l=0, r=0, t=0, b=120),
                showlegend=False,
            )

            st.plotly_chart(fig_global, use_container_width=True, config={"displayModeBar": False})

            a11y_small(
                "Le graphique met en évidence les variables que le modèle utilise le plus souvent "
                "pour l'ensemble des clients, en termes d'importance moyenne absolue des valeurs SHAP."
            )

        else:
            st.info("L'importance globale n'a pas pu être calculée.")

    st.markdown(
        """
        _À gauche : ce qui a le plus influencé la décision pour **ce client**.
        À droite : les variables les plus importantes **en moyenne** pour tous les clients._
        """
    )

# ---------------------------
# Section : Comparaison population
# ---------------------------
elif section == "Comparaison population":
    st.title("Comparaison du client à la population")

    # On part des colonnes numériques
    numeric_cols = df_clients.select_dtypes(include="number").columns.tolist()

    # On exclut l'identifiant client et éventuellement TARGET de la liste
    cols_exclure = [client_id_col]
    if "TARGET" in numeric_cols:
        cols_exclure.append("TARGET")

    numeric_cols = [c for c in numeric_cols if c not in cols_exclure]

    if not numeric_cols:
        st.warning("Aucune variable numérique disponible pour la comparaison.")
    else:
        # Sélecteur de variable, avec nom lisible en français
        feature = st.selectbox(
            "Choisir une caractéristique à comparer :",
            options=sorted(numeric_cols),
            format_func=feature_label,
        )

        # Valeur du client sur cette variable
        client_val = client_row[feature]

        st.subheader(f"Position du client sur : {feature_label(feature)}")

        # --- Série brute pour la population ---
        pop_series = df_clients[feature].dropna().astype(float)

        if pop_series.empty:
            st.warning("Pas assez de données pour afficher cette caractéristique.")
            st.stop()

        # Gestion des outliers : on limite l’affichage aux percentiles 1 % – 99 %
        p1, p99 = np.nanpercentile(pop_series, [1, 99])
        pop_filtered = pop_series[(pop_series >= p1) & (pop_series <= p99)]

        st.write(
            "*Distribution affichée limitée au 1ᵉʳ–99ᵉ percentile pour améliorer la lisibilité (hors valeurs extrêmes).*"
        )

        # Données pour Plotly
        plot_cols = [feature]
        color_col = None

        # Si TARGET est disponible, on l'utilise pour colorer l'histogramme
        if "TARGET" in df_clients.columns:
            color_col = "TARGET"
            plot_cols.append("TARGET")

        plot_df = df_clients[plot_cols].dropna().copy()
        # On applique aussi le filtrage sur les données utilisées pour le graphique
        plot_df = plot_df[(plot_df[feature] >= p1) & (plot_df[feature] <= p99)]

        if color_col is not None:
            plot_df["decision_str"] = np.where(
                plot_df["TARGET"] == 1,
                "Dossiers en défaut",
                "Dossiers remboursés"
            )
            color_arg = "decision_str"
        else:
            color_arg = None

        # --- Histogramme population tronquée ---
        fig_hist = px.histogram(
            plot_df,
            x=feature,
            color=color_arg,
            nbins=40,
            opacity=0.8,
            color_discrete_map={
                "Dossiers en défaut": "#CC0000",
                "Dossiers remboursés": "#0057B8",
            } if color_arg else None,
        )

        fig_hist.update_layout(
            xaxis_title=feature_label(feature),
            yaxis_title="Nombre de clients",
            margin=dict(l=0, r=0, t=10, b=40),
            bargap=0.05,
            legend_title_text="",
        )

        if color_arg:
            a11y_small(
                "En rouge : clients ayant connu un défaut de remboursement. "
                "En bleu : clients ayant remboursé leur crédit."
            )

        # --- Ligne verticale / annotation pour le client ---
        try:
            x_client = float(client_val)

            if p1 <= x_client <= p99:
                # Le client est dans la zone affichée
                fig_hist.add_vline(
                    x=x_client,
                    line_width=3,
                    line_color="red",
                    annotation_text="Client",
                    annotation_position="right",
                    annotation_font_color="red",
                    annotation_font_size=20,
                )
            else:
                # Hors cadre : on ajoute une annotation au bord
                direction = "droite" if x_client > p99 else "gauche"
                fig_hist.add_annotation(
                    x=p99 if x_client > p99 else p1,
                    y=0,
                    text=f"Client hors cadre → ({direction})",
                    showarrow=True,
                    arrowhead=2,
                    yshift=30,
                )
        except Exception:
            pass  # si la valeur n'est pas numérique, on ne met pas de ligne

        st.plotly_chart(
            fig_hist,
            use_container_width=True,
            config={"displayModeBar": False},
        )

        a11y_small(
            f"Le graphique montre la distribution de **{feature_label(feature)}** "
            "pour l'ensemble des clients. La ligne verticale rouge indique la position du client sélectionné."
        )

        # --- Résumé statistique simple sur la distribution complète ---
        if np.isfinite(float(client_val)):
            q10, q50, q90 = np.nanpercentile(pop_series, [10, 50, 90])

            # formatage "français" rapide (espace pour séparateur milliers)
            def fmt(x):
                try:
                    return f"{x:,.2f}".replace(",", " ").replace(".00", "")
                except Exception:
                    return str(x)

            st.markdown(
                f"""
- Valeur du client : **{fmt(client_val)}**
- Médiane de la population : **{fmt(q50)}**
- Intervalle "habituel" (10 % – 90 %) : **[{fmt(q10)} ; {fmt(q90)}]**
                """
            )

            # Position relative du client
            client_val_float = float(client_val)
            if client_val_float < q10:
                pos_txt = "Le client se situe **nettement en dessous** de la majorité des dossiers."
            elif client_val_float > q90:
                pos_txt = "Le client se situe **nettement au-dessus** de la majorité des dossiers."
            elif client_val_float < q50:
                pos_txt = "Le client est **plutôt en dessous** de la moyenne."
            else:
                pos_txt = "Le client est **plutôt au-dessus** de la moyenne."

            a11y_small(
                f"Lecture : {pos_txt} Cela permet au chargé de clientèle de situer ce client par rapport aux autres."
            )
        else:
            a11y_small(
                "Impossible de calculer un résumé statistique pour cette variable (valeur client non numérique)."
            )

# ---------------------------
# Section : Analyse bi-variée
# ---------------------------
elif section == "Analyse bi-variée":
    st.title("Analyse bi-variée")

    # Colonnes numériques
    numeric_cols = df_clients.select_dtypes(include="number").columns.tolist()

    # On exclut l'identifiant client et TARGET des variables candidates
    cols_exclure = [client_id_col]
    if "TARGET" in numeric_cols:
        cols_exclure.append("TARGET")

    numeric_cols = [c for c in numeric_cols if c not in cols_exclure]

    if len(numeric_cols) < 2:
        st.warning("Pas assez de variables numériques pour réaliser une analyse bi-variée.")
    else:
        col_x, col_y = st.columns(2)
        with col_x:
            feature_x = st.selectbox(
                "Variable X",
                options=sorted(numeric_cols),
                key="feature_x",
                format_func=feature_label,
            )
        with col_y:
            feature_y = st.selectbox(
                "Variable Y",
                options=sorted(numeric_cols),
                key="feature_y",
                format_func=feature_label,
            )

        # Option de coloration
        color_options = ["(aucune)"]
        if "TARGET" in df_clients.columns:
            color_options.append("Statut de remboursement (TARGET)")

        color_choice = st.selectbox(
            "Coloration par variable (optionnel)",
            options=color_options,
            index=1 if "Statut de remboursement (TARGET)" in color_options else 0,
        )

        if color_choice == "Statut de remboursement (TARGET)" and "TARGET" in df_clients.columns:
            color_arg = "TARGET"
        else:
            color_arg = None

        # Préparation des données brutes
        cols = [feature_x, feature_y]
        if color_arg is not None:
            cols.append("TARGET")

        df_plot = df_clients[cols].dropna().copy()

        if df_plot.empty:
            st.warning("Pas assez de données disponibles sur ces deux variables.")
        else:
            # Gestion des outliers sur X et Y : 1 % – 99 %
            x_series = df_plot[feature_x].astype(float)
            y_series = df_plot[feature_y].astype(float)

            x1, x99 = np.nanpercentile(x_series, [1, 99])
            y1, y99 = np.nanpercentile(y_series, [1, 99])

            # Filtrage manuel (sans .between, et en s'assurant que c'est 1D)
            mask = (
                (x_series >= x1) & (x_series <= x99) &
                (y_series >= y1) & (y_series <= y99)
            )
            df_plot = df_plot[mask]

            st.write(
                "*Affichage limité au 1ᵉʳ–99ᵉ percentile pour chaque axe afin de réduire l'impact des valeurs extrêmes.*"
            )

            # Construction explicite du DataFrame pour le scatter (1D garanti)
            df_scatter = pd.DataFrame({
                "x": df_plot[feature_x].astype(float).values.ravel(),
                "y": df_plot[feature_y].astype(float).values.ravel(),
            })

            color_plot = None
            if color_arg == "TARGET":
                df_scatter["statut"] = np.where(
                    df_plot["TARGET"].values == 1,
                    "Dossiers en défaut",
                    "Dossiers remboursés",
                )
                color_plot = "statut"

            # Nuage de points population
            fig = plot_bivariate(
                df_clients,
                feature_x,
                feature_y,
                color=color_arg,
                client_row=client_row,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            a11y_small(
                f"Chaque point représente un client dans le plan "
                f"({feature_label(feature_x)} ; {feature_label(feature_y)}). "
                "Le symbole en forme de croix rouge correspond au client sélectionné."
            )


# ---------------------------
# Section : Simulation / Modification client
# ---------------------------
elif section == "Simulation / Modification client":
    st.title("Simulation : Modifier les informations du client")

    st.write(
        "Cette section permet de tester l'impact de changements de caractéristiques "
        "sur la décision du modèle, sans modifier la base réelle."
    )

    if schema is None or len(schema) == 0:
        st.error("Impossible de récupérer le schéma des données depuis l'API.")
        st.stop()

    # On part de la ligne du client actuel
    editable_row = client_row.copy()

    st.subheader("Choix des caractéristiques à modifier")
    a11y_small(
        "Par défaut, seules les caractéristiques les plus importantes pour le modèle "
        "sont proposées. Vous pouvez ajouter d'autres champs si besoin."
    )

    # --- 1) Déterminer les features les plus importantes (global SHAP) ---
    try:
        global_imp = compute_global_importance(
            n_samples=200,
            schema=schema if schema else None,
        )
    except Exception:
        global_imp = None

    if global_imp is not None and not global_imp.empty:
        col_name = "importance" if "importance" in global_imp.columns else global_imp.columns[0]
        important_features = (
            global_imp.sort_values(col_name, ascending=False)
            .head(15)
            .index.tolist()
        )
        # Sécurité : on garde uniquement celles présentes dans le schéma
        important_features = [f for f in important_features if f in schema]
    else:
        # fallback : premières colonnes du schéma (hors ID si besoin)
        important_features = [c for c in schema if c != client_id_col][:15]

    # Liste des autres features possibles à ajouter
    remaining_features = [c for c in schema if c not in important_features]

    # Multi-sélecteur pour ajouter d'autres features
    extra_features = st.multiselect(
        "Ajouter d'autres caractéristiques à la simulation :",
        options=sorted(remaining_features),
        format_func=feature_label,
    )

    # Ensemble final de colonnes éditables (ordre : importantes d'abord)
    editable_cols = important_features + extra_features

    st.markdown("---")
    st.subheader("Modifier les caractéristiques du client")
    a11y_small("⚠️ Ceci ne modifie pas la base réelle — uniquement une simulation locale.")

    with st.form("edit_form"):
        edited_values: dict[str, object] = {}

        # --- 2) Widgets uniquement pour les colonnes choisies ---
        for col in editable_cols:
            raw_val = editable_row[col] if col in editable_row.index else None

            label_fr = feature_label(col)

            # Détection de type simple
            if isinstance(raw_val, (int, np.integer)):
                new_val = st.number_input(
                    label_fr,
                    value=int(raw_val) if raw_val is not None else 0,
                )
            elif isinstance(raw_val, (float, np.floating)):
                new_val = st.number_input(
                    label_fr,
                    value=float(raw_val) if raw_val is not None else 0.0,
                    step=0.1,
                )
            else:
                new_val = st.text_input(
                    label_fr,
                    value=str(raw_val) if raw_val is not None else "",
                )

            edited_values[col] = new_val

        submitted = st.form_submit_button("Simuler la décision")

    if submitted:
        st.subheader("Résultat de la simulation")

        # --- 3) Construire le payload complet pour l'API ---
        # Pour toutes les colonnes du schéma :
        # - si l'utilisateur a modifié la valeur (editable_cols) => on prend edited_values
        # - sinon => on garde la valeur originale du client
        features_sim: dict[str, object] = {}
        for col in schema:
            if col in edited_values:
                v = edited_values[col]
            else:
                v = editable_row[col] if col in editable_row.index else None
            features_sim[col] = to_json_serializable(v)

        # --- 4) Appels API /predict et /explain avec les valeurs simulées ---
        try:
            pred_sim = predict(features_sim)
            proba_sim = float(pred_sim.get("probability", 0.0))
            pred_label_sim = "ACCORDÉ" if pred_sim.get("prediction", 0) == 0 else "REFUSÉ"

            st.metric("Décision simulée", pred_label_sim)
            st.metric("Probabilité de défaut simulée", f"{proba_sim * 100:.2f} %")
        except Exception as e:
            st.error(f"Erreur lors de l'appel à l'API /predict: {e}")
            st.stop()

        st.markdown("---")
        st.subheader("Variables impactant la décision simulée (SHAP)")

        try:
            explain_sim = explain(features_sim)
            contrib_sim = explain_sim.get("contrib", {})
        except Exception as e:
            st.error(f"Erreur lors de l'appel à l'API /explain: {e}")
            contrib_sim = {}

        if contrib_sim:
            df_sim_local = (
                pd.DataFrame.from_dict(contrib_sim, orient="index", columns=["shap_value"])
                .assign(abs_shap=lambda d: d["shap_value"].abs())
                .sort_values("abs_shap", ascending=False)
                .head(15)
            )

            df_sim_local["label"] = df_sim_local.index.map(feature_label)

            fig_sim = px.bar(
                df_sim_local.reset_index(drop=True),
                x="label",
                y="shap_value",
                color="shap_value",
                color_continuous_scale=["#0057B8", "#CC0000"],  # bleu -> rouge (WCAG ok)
            )
            fig_sim.update_layout(
                xaxis_title="Variables",
                yaxis_title="Impact SHAP (simulation)",
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig_sim, use_container_width=True)

            a11y_small(
                "Les barres indiquent les variables qui ont le plus influencé la décision simulée : "
                "les valeurs positives augmentent le risque estimé, les valeurs négatives le diminuent."
            )
        else:
            st.info("Aucune importance locale disponible pour cette simulation.")
