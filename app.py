import numpy as np
import pandas as pd
import streamlit as st

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
)

# ---------------------------
# Config de la page
# ---------------------------
st.set_page_config(
    page_title="Dashboard scoring crédit - Projet 8",
    layout="wide"
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
        st.caption(str(e))

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
        "Analyse bi-variée"
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
            decision_label = "ACCORD" if prediction == 0 else "REFUS"
            st.metric("Décision modèle", decision_label)
        else:
            st.metric("Décision modèle", "Indisponible")

    # --- Carte 2 : Probabilité & niveau de risque ---
    with col2:
        st.subheader("Niveau de risque")
        if proba is not None:
            proba_pct = proba * 100

            if proba < 0.10:
                risk_level = "Faible"
                risk_expl = "Le risque estimé est faible."
            elif proba < 0.30:
                risk_level = "Modéré"
                risk_expl = "Le risque estimé est modéré."
            else:
                risk_level = "Élevé"
                risk_expl = "Le risque estimé est élevé."

            st.metric("Probabilité de défaut", f"{proba_pct:.1f} %")
            st.write(f"Niveau de risque : **{risk_level}**")
            st.caption(risk_expl)
            st.progress(min(max(proba, 0.0), 1.0))
        else:
            st.write("Probabilité de défaut indisponible.")

    # --- Carte 3 : Seuil & distance ---
    with col3:
        st.subheader("Seuil de décision")
        if proba is not None:
            distance = abs(proba - threshold)
            st.write(f"Seuil de décision : **{threshold:.0%}**")
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
        st.caption("Résumé indisponible car la prédiction n'a pas pu être calculée.")

    st.markdown("---")
    st.subheader("Caractéristiques principales du client")

    st.dataframe(
        pd.DataFrame(client_row).T,
        use_container_width=True
    )

# ---------------------------
# Section : Interprétation du score (SHAP local)
# ---------------------------
elif section == "Interprétation du score":
    st.title("Interprétation du score (importance locale)")

    try:
        explain_result = explain(features)
        base_value = explain_result.get("base_value", None)
        contrib = explain_result.get("contrib", {})
    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API /explain: {e}")
        contrib = {}
        base_value = None

    if contrib:
        shap_df = (
            pd.DataFrame.from_dict(contrib, orient="index", columns=["shap_value"])
            .sort_values("shap_value", key=lambda s: s.abs(), ascending=False)
        )

        st.markdown("### Top variables qui influencent la décision pour ce client")
        st.bar_chart(shap_df)

        st.caption(
            "Les valeurs SHAP (en bleu) indiquent la contribution de chaque variable à "
            "l'augmentation ou la diminution du risque estimé pour ce client."
        )

        if base_value is not None:
            st.caption(f"Valeur de base du modèle (base_value) : `{base_value:.4f}`")
    else:
        st.info(
            "Les contributions SHAP ne sont pas disponibles. "
            "Vérifiez que l'endpoint `/explain` est fonctionnel sur l'API."
        )

# ---------------------------
# Section : Comparaison population
# ---------------------------
elif section == "Comparaison population":
    st.title("Comparaison du client à la population")

    numeric_cols = df_clients.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        st.warning("Aucune variable numérique disponible pour la comparaison.")
    else:
        feature = st.selectbox("Choisir une variable à comparer", options=numeric_cols)

        try:
            fig = plot_client_vs_population(df_clients, client_row, feature)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur lors de la génération du graphique: {e}")

# ---------------------------
# Section : Analyse bi-variée
# ---------------------------
elif section == "Analyse bi-variée":
    st.title("Analyse bi-variée")

    numeric_cols = df_clients.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Pas assez de variables numériques pour réaliser une analyse bi-variée.")
    else:
        col_x, col_y = st.columns(2)
        with col_x:
            feature_x = st.selectbox("Variable X", options=numeric_cols, key="feature_x")
        with col_y:
            feature_y = st.selectbox("Variable Y", options=numeric_cols, key="feature_y")

        color_col = st.selectbox(
            "Coloration par variable (optionnel)",
            options=["(aucune)"] + numeric_cols,
            index=0
        )
        color_arg = None if color_col == "(aucune)" else color_col

        try:
            fig = plot_bivariate(df_clients, feature_x, feature_y, color=color_arg)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur lors de la génération du graphique bi-varié: {e}")
