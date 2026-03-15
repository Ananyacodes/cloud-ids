from __future__ import annotations
import logging
import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference.serving.predictor import Predictor
from ingestion.parsers import FlowRecord
from monitoring.metrics import INFERENCE_LATENCY, DECISION_COUNTER, REQUEST_COUNTER

logger = logging.getLogger(__name__)

_predictor: Predictor | None = None
_router = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor, _router
    if _predictor is None:
        logger.info("Loading models…")
        _predictor = Predictor()
    if _router is None:
        from triage.router import TriageRouter

        _router = TriageRouter()
    logger.info("Models ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Cloud IDS Inference API", version="1.0.0", lifespan=lifespan)


class FlowRequest(BaseModel):
    flow_id: str
    timestamp: str = ""
    traffic_source: str = "netflow"
    duration: float = 0.0
    bytes_fwd: int = 0
    bytes_bwd: int = 0
    packets_fwd: int = 0
    packets_bwd: int = 0
    pkt_size_mean: float = 0.0
    pkt_size_std: float = 0.0
    iat_mean: float = 0.0
    iat_std: float = 0.0
    fin_flag_cnt: int = 0
    syn_flag_cnt: int = 0
    rst_flag_cnt: int = 0
    psh_flag_cnt: int = 0
    ack_flag_cnt: int = 0
    flow_bytes_per_sec: float = 0.0
    flow_pkts_per_sec: float = 0.0
    protocol: str = "TCP"
    src_port_bucket: str = "ephemeral"
    dst_port_bucket: str = "unknown"
    ja3_hash: str = ""
    ja3s_hash: str = ""
    sni_entropy: float = 0.0
    cert_validity_days: int = -1
    tls_version: str = ""


class PredictResponse(BaseModel):
    flow_id: str
    ensemble_score: float
    lstm_score: float
    xgb_score: float
    ae_score: float
    decision: str
    routed: bool = False


class BatchRequest(BaseModel):
    flows: List[FlowRequest] = Field(..., max_length=512)


class BatchResponse(BaseModel):
    results: List[PredictResponse]
    latency_ms: float



@app.get("/healthz")
async def health():
    return {"status": "ok"}


@app.get("/readyz")
async def ready():
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "ready"}


@app.post("/predict", response_model=PredictResponse)
async def predict_one(req: FlowRequest):
    if _predictor is None or _router is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    REQUEST_COUNTER.labels(endpoint="predict").inc()
    t0 = time.perf_counter()
    record = FlowRecord(**req.model_dump())
    result = _predictor.predict_one(record)
    routed = _router.route(result)
    latency = (time.perf_counter() - t0) * 1000
    INFERENCE_LATENCY.observe(latency)
    DECISION_COUNTER.labels(decision=result.decision.value).inc()
    return PredictResponse(
        flow_id=result.flow_id,
        ensemble_score=result.ensemble_score,
        lstm_score=result.lstm_score,
        xgb_score=result.xgb_score,
        ae_score=result.ae_score,
        decision=result.decision.value,
        routed=routed,
    )


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(req: BatchRequest):
    if _predictor is None or _router is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    REQUEST_COUNTER.labels(endpoint="predict_batch").inc()
    t0 = time.perf_counter()
    records = [FlowRecord(**f.model_dump()) for f in req.flows]
    results = _predictor.predict_batch(records)
    routed_flags = [_router.route(r) for r in results]
    latency = (time.perf_counter() - t0) * 1000
    INFERENCE_LATENCY.observe(latency)
    for r in results:
        DECISION_COUNTER.labels(decision=r.decision.value).inc()
    return BatchResponse(
        results=[
            PredictResponse(
                flow_id=r.flow_id,
                ensemble_score=r.ensemble_score,
                lstm_score=r.lstm_score,
                xgb_score=r.xgb_score,
                ae_score=r.ae_score,
                decision=r.decision.value,
                routed=flag,
            )
            for r, flag in zip(results, routed_flags)
        ],
        latency_ms=latency,
    )


from monitoring.metrics import metrics_app
app.mount("/metrics", metrics_app)
