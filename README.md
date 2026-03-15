# Cloud IDS — ML-Based Network Intrusion Detection System

GCP-native, Python 3.11 + PyTorch, Docker/Kubernetes deployment.

## Architecture

Three parallel detectors → confidence arbitration → asymmetric triage:

- **LSTM classifier** — sequential/temporal attacks (slow brute-force, exfiltration)
- **XGBoost classifier** — known attack signatures (fast, precise)
- **Autoencoder** — zero-day anomaly detection (unsupervised)
- **Ensemble arbitration** — soft-vote with asymmetric thresholds (low=0.35, high=0.75)
- **Analyst queue** — human-in-the-loop for 0.35–0.75 band
- **Retraining loop** — analyst verdicts feed back into model updates

## Quick Start

```bash
pip install -r requirements.txt
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
cp configs/config.example.yaml configs/config.yaml
python scripts/train_all.py --config configs/config.yaml
docker-compose -f deployment/docker/docker-compose.yml up
```

## Project Structure

```
cloud-ids/
├── configs/               YAML config files
├── data/                  Raw, processed, sample data
├── ingestion/             Pub/Sub consumer, parsers
├── features/              Feature extraction, preprocessing
├── models/
│   ├── lstm/              LSTM sequence classifier
│   ├── xgboost/           XGBoost signature classifier
│   ├── autoencoder/       Anomaly detection autoencoder
│   └── ensemble/          Soft-voting ensemble + calibration
├── training/              Training pipelines
├── inference/api/         FastAPI inference service
├── triage/                Threshold engine, alert routing
├── retraining/            Drift detection, retraining scheduler
├── deployment/            Docker, K8s, GCP configs
├── monitoring/            Prometheus metrics, drift alerts
└── tests/                 Unit + integration tests
```
