"""Central configuration. Single source of truth for env-driven settings
and the feature vocabulary shared by training and serving."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# Train/serve parity contract: order and membership must never drift.
FEATURE_COLUMNS: list[str] = [
    "amt", "log_amt", "hour", "dow", "is_weekend", "is_night",
    "cc_cnt_1h", "cc_amt_1h", "cc_cnt_24h", "cc_amt_24h",
    "cc_time_since_prev_s", "cc_amt_mean_7d", "cc_amt_std_7d",
    "cc_amt_zscore", "cc_distinct_cat_7d",
]


class Settings(BaseSettings):
    """Runtime settings; every field overridable via environment variable."""
    model_config = SettingsConfigDict(env_prefix="FRAUD_", extra="ignore")

    region: str = "ca-central-1"

    # S3 layout.
    s3_bucket: str = "fraudds-bucket"
    data_key: str = "data/fraudTrain.csv"
    model_key: str = "models/model.txt"
    calibrator_key: str = "models/calibrator.pkl"
    predictions_prefix: str = "predictions/"

    # DynamoDB online feature store.
    ddb_table: str = "card-state"
    shard_count: int = 16          # spreads write throughput across partitions
    state_history_days: int = 7    # window retained per card for 7d features
    state_ttl_days: int = 10       # TTL horizon; > history so deletes never race reads

    # Decision thresholds (calibrated probability).
    review_threshold: float = 0.30
    decline_threshold: float = 0.70


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so config is parsed once per process."""
    return Settings()
