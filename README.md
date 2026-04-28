# RenderCloud — Multi-Disease Risk Prediction System

A machine learning-powered health risk assessment platform that predicts the likelihood of chronic diseases (Diabetes, Hypertension, CVD, CKD) from patient features, with SHAP-based explanations and population-level comparisons.

## Features

- **Multi-disease prediction** — Diabetes, Hypertension, Cardiovascular Disease, and Chronic Kidney Disease
- **Dual model versions** — `basic` (fewer inputs) and `full` (complete feature set) per disease
- **SHAP interpretability** — Explains which features drive each prediction
- **Population comparison** — Benchmarks user risk against age/gender-matched NHANES cohorts
- **ONNX + joblib support** — Automatic fallback between model formats
- **Microservice-ready** — Runs monolithically or via Docker Compose

## Architecture

```
RenderCloud/
├── backend/              # FastAPI REST API (port 8000)
│   ├── app.py            # Application entry point
│   ├── config_.py        # Model registry & feature definitions
│   ├── core_services/    # Prediction, metrics, training logic
│   ├── routers/          # API route handlers
│   └── models/           # Pre-trained model files
├── frontend/             # Streamlit UI (port 8501)
│   └── app.py
└── data/                 # NHANES dataset & population stats

Docker_Method/            # Microservice deployment
├── docker-compose.yml
├── gateway/              # API gateway (port 8080)
├── ckd-service/          # CKD microservice (port 5001)
└── frontend/             # Dockerised Streamlit
```

## Quick Start

### Local (monolithic)

```bash
# Backend
cd RenderCloud/backend
pip install -r requirements.txt
python app.py
# → http://localhost:8000

# Frontend (separate terminal)
cd RenderCloud/frontend
streamlit run app.py
# → http://localhost:8501
```

### Docker (microservices)

```bash
cd Docker_Method
docker-compose up
# Gateway  → http://localhost:8080
# Frontend → http://localhost:8501
# CKD svc  → http://localhost:5001
```

## API Reference

### Prediction

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/prediction/{disease}` | Single-disease prediction |
| POST | `/prediction/all` | All diseases in one call |

`{disease}` — `diabetes` · `hypertension` · `cvd` · `ckd` · `all`

**Request body:**
```json
{
  "input_data": {
    "age": 55,
    "bmi": 28.4,
    "glucose": 110
  }
}
```

Optional query param: `?model_version=basic` (default: `full`)

**Response:**
```json
{
  "disease": "diabetes",
  "risk_probability": 0.34,
  "risk_level": "moderate",
  "population_comparison": { ... }
}
```

### Metrics / Interpretability

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/metrics/{metric_type}` | Single metric (`shap`, `feature_importance`, `confidence`, `performance`) |
| POST | `/metrics/batch` | Multiple metrics in one call |
| GET | `/metrics/available` | List supported metric types |
| POST | `/metrics/cache/clear` | Flush model cache |

### Utility

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome |
| GET | `/health` | Health check → `{"status": "ok"}` |

## Models

| Disease | Formats | Model type |
|---------|---------|-----------|
| Diabetes | `.joblib` / `.onnx` | Scikit-learn / ONNX |
| Hypertension | `.joblib` / `.onnx` | Scikit-learn / ONNX |
| CVD | `.joblib` / `.onnx` | Scikit-learn / ONNX |
| CKD | `.pkl` | LightGBM with mask |

Training data: NHANES 2021–2023 (`RenderCloud/data/nhanes_2021_2023_master.csv`)

## Dependencies

Key packages (see `RenderCloud/backend/requirements.txt`):

```
fastapi  uvicorn  pydantic
scikit-learn  lightgbm  xgboost
onnxruntime  shap  joblib
pandas  numpy  requests  gunicorn
```

Frontend: `streamlit`

## Production

The backend is deployed on Render Cloud. To run with Gunicorn locally:

```bash
gunicorn RenderCloud.backend.app:app -w 4 -k uvicorn.workers.UvicornWorker
```
