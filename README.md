# Gateway ML Studio

Auto ML control plane that:
- Detects task type (classification, regression, anomaly)
- Trains multiple models per task
- Routes traffic with a bandit based on live performance
- Monitors SLOs and triggers retraining on drift

## Tech stack

- Python, pandas, numpy, scikit learn, XGBoost
- Streamlit or FastAPI for UI/API
- MLflow or custom artifact logging
- Docker ready (optional in future)

## Architecture

- Ingestion layer: loads CSV or DB data
- Task detector: inspects target column to infer problem type
- Trainer: trains a model zoo (RF, XGBoost, linear, etc)
- Bandit router: routes requests to the best model
- Monitor: tracks metrics, drift, and SLOs

## Quickstart

```bash
git clone https://github.com/Anirudh2141-DS/Gateway-ML-Studio.git
cd Gateway-ML-Studio
pip install -r requirements.txt
streamlit run app.py
