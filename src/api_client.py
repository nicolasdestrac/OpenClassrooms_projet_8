import os
import requests

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://openclassrooms-projet7-scoring-api.onrender.com/"
)

def _post(endpoint: str, payload: dict) -> dict:
    """Helper générique POST."""
    url = f"{API_BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    resp = requests.post(url, json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()

def _get(endpoint: str) -> dict:
    """Helper générique GET."""
    url = f"{API_BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()

# -------------------------------------------------
# Appels dédiés à ton API de scoring
# -------------------------------------------------

def get_schema() -> list:
    """
    Récupère les colonnes attendues par le modèle via /schema.
    """
    data = _get("/schema")
    return data.get("input_columns", [])

def predict_proba(features: dict) -> float:
    """
    Appelle /predict_proba et renvoie la probabilité.
    """
    data = _post("/predict_proba", {"features": features})
    return float(data["probability"])

def predict(features: dict) -> dict:
    """
    Appelle /predict et renvoie :
    {
        "probability": float,
        "prediction": 0/1,
        "threshold": float
    }
    """
    return _post("/predict", {"features": features})

def explain(features: dict) -> dict:
    """
    Appelle /explain et renvoie :
    {
      "base_value": float,
      "contrib": {feature: shap_value, ...}
    }
    """
    return _post("/explain", {"features": features})

def check_health() -> dict:
    """
    Vérifie que l'API fonctionne via /health.
    """
    return _get("/health")
