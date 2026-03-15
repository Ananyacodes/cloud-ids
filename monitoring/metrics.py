from __future__ import annotations
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app

INFERENCE_LATENCY = Histogram(
    "ids_inference_latency_ms",
    "End-to-end inference latency in milliseconds",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
)

DECISION_COUNTER = Counter(
    "ids_decisions_total",
    "Total decisions by outcome",
    labelnames=["decision"],
)

REQUEST_COUNTER = Counter(
    "ids_requests_total",
    "Total prediction requests",
    labelnames=["endpoint"],
)

RECONSTRUCTION_ERROR_GAUGE = Gauge(
    "ids_ae_reconstruction_error_mean",
    "Rolling mean reconstruction error on live traffic (anomaly drift signal)",
)

metrics_app = make_asgi_app()
