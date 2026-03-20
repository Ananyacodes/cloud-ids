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

## Live Traffic Scanning

The sample flows are only for smoke testing. For real packet capture and scoring,
run the live scanner:

```bash
# List interfaces first (Windows PowerShell)
python -c "from scapy.all import get_if_list; print('\\n'.join(get_if_list()))"

# Capture live traffic and run through IDS models
python scripts/scan_live.py --iface "Ethernet" --bpf "ip" --flow-idle-timeout 5

# Dry-run mode: capture + feature extraction only (no model inference)
python scripts/scan_live.py --iface "Ethernet" --dry-run

# Publish captured real flows to Pub/Sub ingestion topic
python scripts/scan_live.py --iface "Ethernet" --publish-pubsub

# Shadow mode: run local inference and publish the same flows to Pub/Sub
python scripts/scan_live.py --iface "Ethernet" --shadow-pubsub
```

Notes:
- Run with Administrator/root privileges so packet capture works.
- On Windows, install Npcap if packet capture is unavailable.
- Use narrower BPF filters in production, e.g. `--bpf "tcp port 443"`.

## CLI

Use the unified CLI:

```bash
# List interfaces
python -m scripts.cli interfaces

# Local inference mode
python -m scripts.cli scan --iface "Ethernet" --mode local

# Dry-run mode
python -m scripts.cli scan --iface "Ethernet" --mode dry-run

# Publish mode
python -m scripts.cli scan --iface "Ethernet" --mode publish

# Shadow mode (local inference + publish)
python -m scripts.cli scan --iface "Ethernet" --mode shadow

# No-GCS local mode: use local artifacts only
python -m scripts.bootstrap_local_artifacts
set IDS_DISABLE_GCS=1
set IDS_DISABLE_PUBSUB=1
python -m scripts.cli scan --mode local
```

No-GCS requirements:
- Place these files in `artifacts/` (or set `models.artifacts_dir` to a local folder):
	`preprocessor.pkl`, `xgboost_model.pkl`, `lstm_best.pt`, `autoencoder_final.pt`.

## Run Commands

Windows PowerShell commands for the most common run modes:

```powershell
# 1. List capture interfaces
python -m scripts.cli interfaces

# 2. Dry-run live capture with visible flow logs
python -m scripts.cli scan --mode dry-run --flow-idle-timeout 1 --bpf ip

# 3. Offline local inference with visible score/decision logs
python -m scripts.bootstrap_local_artifacts
$env:IDS_DISABLE_GCS = '1'
$env:IDS_DISABLE_PUBSUB = '1'
$env:IDS_REFRESH_LOCAL_ARTIFACTS = '1'
python -m scripts.cli scan --mode local --flow-idle-timeout 1 --bpf ip

# 4. Publish live flows to Pub/Sub
$env:GOOGLE_APPLICATION_CREDENTIALS = 'C:\path\to\service-account.json'
python -m scripts.cli scan --mode publish --flow-idle-timeout 1 --bpf ip

# 5. Shadow mode: local inference + Pub/Sub publish
$env:GOOGLE_APPLICATION_CREDENTIALS = 'C:\path\to\service-account.json'
python -m scripts.cli scan --mode shadow --flow-idle-timeout 1 --bpf ip
```

If no live logs appear immediately, generate traffic in another terminal:

```powershell
ping -n 5 1.1.1.1
nslookup github.com
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
