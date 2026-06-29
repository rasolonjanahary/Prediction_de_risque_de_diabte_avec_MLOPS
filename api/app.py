from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request, Response, Form
from pydantic import BaseModel
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,  # 🆕 Ajouté pour suivre le Data Drift
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from pathlib import Path
import pandas as pd
import joblib
import time
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI()

templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))

model = joblib.load(BASE_DIR / "models" / "diabete_model_svc.pkl")
std = joblib.load(BASE_DIR / "models" / "scaler.pkl")

# ---MÉTRIQUES ---
PREDICTION_COUNTER = Counter(
    "diabetes_prediction_total",
    "Nombre total de prédictions"
)
POSITIVE_PREDICTION_COUNTER = Counter(
    "diabetes_positive_prediction_total",
    "Nombre de prédictions positives"
)
PREDICTION_DURATION = Histogram(
    "diabetes_prediction_duration_seconds",
    "Temps d'exécutions des prédictions"
)
ERROR_COUNTER = Counter(
    "diabetes_prediction_errors_total",
    "Nombre d'erreurs de prédiction"
)

# 1. Suivi HTTP global (permet de voir les erreurs 4xx de validation Pydantic)
HTTP_REQUESTS_COUNTER = Counter(
    "http_requests_total",
    "Total des requêtes HTTP reçues",
    ["method", "endpoint", "http_status"]
)

# 2. Data Drift Gauges (surveillent les moyennes mouvantes des features clés)
DRIFT_GLUCOSE_GAUGE = Gauge(
    "input_glucose_average",
    "Moyenne du taux de glucose des derniers patients"
)
DRIFT_BMI_GAUGE = Gauge(
    "input_bmi_average",
    "Moyenne de l'IMC (BMI) des derniers patients"
)
DRIFT_INSULIN_GAUGE = Gauge(
    "input_insulin_average",
    "Moyenne de l'insulin des derniers patients"
)

DRIFT_TRICEPS_GAUGE = Gauge(
    "input_triceps_average",
    "Moyenne de triceps des derniers patients"
)


class Patient(BaseModel):
    pregnancies: int
    glucose: float
    diastolic: float
    triceps: float
    insulin: float
    bmi: float
    dpf: float
    age: int


# Middleware pour intercepter et logger automatiquement le trafic HTTP
@app.middleware("http")
async def monitor_http_traffic(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    # Exclure l'endpoint /metrics des logs pour éviter de polluer Prometheus
    if request.url.path != "/metrics":
        HTTP_REQUESTS_COUNTER.labels(
            method=request.method,
            endpoint=request.url.path,
            http_status=response.status_code
        ).inc()
        
    return response


@app.get("/")
def accueil(request: Request):
    return templates.TemplateResponse(request=request, name="accueil.html")


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: Patient):
    try:
        start_time = time.time()
        dt = data.model_dump()

        # 🆕 Mise à jour des indicateurs de dérive de données (Data Drift)
        DRIFT_GLUCOSE_GAUGE.set(dt["glucose"])
        DRIFT_BMI_GAUGE.set(dt["bmi"])
        DRIFT_INSULIN_GAUGE.set(dt["insulin"])
        DRIFT_TRICEPS_GAUGE.set(dt["triceps"])

        df = pd.DataFrame([dt])
        data_std = std.transform(df)
        prediction = model.predict(data_std)
        prediction_value = int(prediction[0])

        PREDICTION_COUNTER.inc()

        if prediction_value == 1:
            POSITIVE_PREDICTION_COUNTER.inc()

        duration = time.time() - start_time
        PREDICTION_DURATION.observe(duration)

        return {
            "resultat": prediction_value,
            "temps_prediction": round(duration, 5),
            "patient": dt
        }
    except Exception as e:
        ERROR_COUNTER.inc()
        return {"error": str(e)}
