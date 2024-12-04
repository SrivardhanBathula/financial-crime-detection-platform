"""
FastAPI Application — Financial Crime Detection Platform
Serves real-time fraud scoring, risk assessment, and investigation endpoints.
99.9% availability SLA with async inference and health monitoring.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import PlainTextResponse

from src.models.fraud_detector import FraudDetectionModel
from src.models.risk_scoring import RiskScoringEngine

logger = logging.getLogger(__name__)

# ─── Prometheus Metrics ───────────────────────────────────────────────────────
REQUEST_COUNT = Counter("fraud_api_requests_total", "Total API requests", ["endpoint", "status"])
INFERENCE_LATENCY = Histogram("fraud_inference_latency_seconds", "Model inference latency", ["model"])
FRAUD_SCORE_HISTOGRAM = Histogram("fraud_score_distribution", "Distribution of fraud scores",
                                   buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# ─── Models (loaded at startup) ───────────────────────────────────────────────
fraud_model: Optional[FraudDetectionModel] = None
risk_engine: Optional[RiskScoringEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models on startup; clean up on shutdown."""
    global fraud_model, risk_engine
    logger.info("Loading fraud detection model...")
    fraud_model = FraudDetectionModel.load("models/fraud_detector")
    risk_engine = RiskScoringEngine.load("models/risk_engine")
    logger.info("Models loaded successfully.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Financial Crime Detection API",
    description="Real-time fraud scoring and risk analytics platform — State Street AI/ML",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type"],
)

security = HTTPBearer()


# ─── Request / Response Schemas ───────────────────────────────────────────────

class TransactionFeatures(BaseModel):
    transaction_id: str
    account_id: str
    txn_count_1h: float = Field(..., ge=0)
    txn_count_24h: float = Field(..., ge=0)
    txn_count_7d: float = Field(..., ge=0)
    txn_amount_sum_1h: float
    txn_amount_sum_24h: float
    txn_amount_avg_7d: float
    txn_amount_stddev_7d: float
    amount_zscore: float
    log_amount: float
    is_first_merchant_visit: int = Field(..., ge=0, le=1)
    is_cross_border: int = Field(..., ge=0, le=1)
    amount_vs_merchant_avg: float
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=1, le=7)
    is_weekend: int = Field(..., ge=0, le=1)
    is_night_txn: int = Field(..., ge=0, le=1)
    seconds_since_last_txn: Optional[float] = None
    is_round_amount: int = Field(..., ge=0, le=1)


class FraudScoreResponse(BaseModel):
    transaction_id: str
    account_id: str
    fraud_probability: float
    is_fraud: bool
    risk_score: float
    risk_tier: str  # LOW / MEDIUM / HIGH / CRITICAL
    inference_time_ms: float
    model_version: str


class BatchScoringRequest(BaseModel):
    transactions: list[TransactionFeatures] = Field(..., max_items=1000)


class BatchScoringResponse(BaseModel):
    results: list[FraudScoreResponse]
    total_processed: int
    flagged_count: int
    processing_time_ms: float


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_risk_tier(score: float) -> str:
    if score < 25:   return "LOW"
    if score < 50:   return "MEDIUM"
    if score < 75:   return "HIGH"
    return "CRITICAL"

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    # Replace with actual JWT verification in production
    if not credentials.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return credentials.credentials


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": fraud_model is not None}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return generate_latest()


@app.post("/v1/score/transaction", response_model=FraudScoreResponse)
async def score_transaction(
    transaction: TransactionFeatures,
    token: str = Depends(verify_token),
):
    """Score a single transaction for fraud probability in real-time."""
    import pandas as pd
    start = time.perf_counter()

    df = pd.DataFrame([transaction.model_dump()])
    with INFERENCE_LATENCY.labels(model="fraud_detector").time():
        fraud_prob = float(fraud_model.predict_proba(df)[0])
        risk_score = float(risk_engine.score(df)[0])

    elapsed_ms = (time.perf_counter() - start) * 1000
    FRAUD_SCORE_HISTOGRAM.observe(fraud_prob)
    REQUEST_COUNT.labels(endpoint="/v1/score/transaction", status="200").inc()

    return FraudScoreResponse(
        transaction_id=transaction.transaction_id,
        account_id=transaction.account_id,
        fraud_probability=round(fraud_prob, 4),
        is_fraud=fraud_prob >= fraud_model.threshold,
        risk_score=round(risk_score, 2),
        risk_tier=get_risk_tier(risk_score),
        inference_time_ms=round(elapsed_ms, 2),
        model_version="xgb-v1.0",
    )


@app.post("/v1/score/batch", response_model=BatchScoringResponse)
async def score_batch(
    request: BatchScoringRequest,
    token: str = Depends(verify_token),
):
    """Score a batch of transactions (up to 1,000) efficiently."""
    import pandas as pd
    start = time.perf_counter()

    df = pd.DataFrame([t.model_dump() for t in request.transactions])
    fraud_probs = fraud_model.predict_proba(df)
    risk_scores = risk_engine.score(df)

    results = []
    for i, txn in enumerate(request.transactions):
        prob = float(fraud_probs[i])
        score = float(risk_scores[i])
        results.append(FraudScoreResponse(
            transaction_id=txn.transaction_id,
            account_id=txn.account_id,
            fraud_probability=round(prob, 4),
            is_fraud=prob >= fraud_model.threshold,
            risk_score=round(score, 2),
            risk_tier=get_risk_tier(score),
            inference_time_ms=0,
            model_version="xgb-v1.0",
        ))

    elapsed_ms = (time.perf_counter() - start) * 1000
    flagged = sum(1 for r in results if r.is_fraud)
    REQUEST_COUNT.labels(endpoint="/v1/score/batch", status="200").inc()

    return BatchScoringResponse(
        results=results,
        total_processed=len(results),
        flagged_count=flagged,
        processing_time_ms=round(elapsed_ms, 2),
    )


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False, workers=4)
