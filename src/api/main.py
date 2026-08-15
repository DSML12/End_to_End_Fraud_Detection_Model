"""FastAPI entrypoint (ECS). Thin: validate input, assign the timestamp,
delegate to the shared inference core, shape the response for the UI."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference_pipeline.predict import Transaction, handle_transaction

app = FastAPI(title="Fraud Detection API")


class PredictRequest(BaseModel):
    """Client payload. No timestamp — the server assigns it at ingest."""
    cc_num: str = Field(min_length=12, max_length=19, pattern=r"^\d+$")
    amt: float = Field(gt=0)
    category: str = Field(min_length=1)


class PredictResponse(BaseModel):
    fraud_score: float
    decision: str
    cc_cnt_1h: float
    cc_cnt_24h: float


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe for ECS/ALB health checks."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    d = handle_transaction(Transaction(req.cc_num, req.amt, req.category))
    return PredictResponse(
        fraud_score=d.fraud_score,
        decision=d.decision,
        cc_cnt_1h=d.cc_cnt_1h,
        cc_cnt_24h=d.cc_cnt_24h,
    )
