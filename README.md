# End-to-End Fraud Detection Model

Real-time credit-card fraud scoring model using both static and streaming features depolyed on ECS Cluster. A LightGBM model with isotonic calibration serves transaction-level fraud scores over a FastAPI service, backed by a DynamoDB online feature store that maintains velocity features across transactions.


## Architecture

```

UI ──▶ API (FastAPI/ECS) ──▶ Predictor
                                ├─ DynamoDB  (online feature store, sync)
                                ├─ S3        (model + calibrator, predictions)
                                └─ LightGBM + isotonic calibrator

Kinesis stream ──▶ Lambda ──▶ Predictor   

```
Per-card state is committed to DynamoDB synchronously before the next transaction is scored, so velocity features (cc_cnt_1h, cc_cnt_24h, …) reflect true history. Predictions are written to S3 asynchronously.

## Live Demo

[End-to-End Fraud Detection Model Live Demo](https://github.com/user-attachments/assets/d96ce9d9-66b8-4b19-bac0-f0ff25a62d3a)

## Layout

```
src/
  config.py              
  features.py            
  storage.py             
  metrics.py            
  api/main.py            
  lambda_handler.py      
  feature_pipeline/      
  train_pipeline/        
  inference_pipeline/    
  monitoring/           
tests/                   
docker/                  
```

## Setup

Requires Python 3.11 and AWS credentials with S3 + DynamoDB access.

```bash
pip install -e ".[api,ui,dev]"
```

## Training

```bash
python -m src.feature_pipeline.run    # S3 raw CSV → matrix.parquet
python -m src.train_pipeline.run      # model.txt, calibrator.pkl, reference.json
```

## Serving

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
streamlit run app.py                  # optional demo UI
```

## Monitoring

Three jobs, split by whether ground truth has matured:

```bash
python -m src.monitoring.run_unlabeled   # PSI/CSI + population drift — no labels needed
python -m src.monitoring.run_labeled     # PR-AUC control chart + calibration — needs labels
python -m src.monitoring.remediate       # applies the policy, triggers retraining
```

## Configuration

Settings are environment-driven with the FRAUD_ prefix (see src/config.py):

| Variable | Default |
|---|---|
| `FRAUD_REGION` | `ca-central-1` |
| `FRAUD_S3_BUCKET` | `fraudds-bucket` |
| `FRAUD_DDB_TABLE` | `card-state` |
| `FRAUD_SHARD_COUNT` | `16` |
| `FRAUD_STATE_HISTORY_DAYS` | `7` |
| `FRAUD_STATE_TTL_DAYS` | `10` |


## Tests

```bash
pytest                    
ruff check src/ tests/
```


## Deploy

CI/CD (.github/workflows/ci_cd.yml) builds both images, pushes to ECR, and redeploys the ECS service and Lambda on push to main. Infrastructure (DynamoDB table, Kinesis stream, Lambda, ECS Cluster) is provisioned once; the pipeline handles subsequent deploys.


