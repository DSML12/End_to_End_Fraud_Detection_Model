"""CloudWatch EMF emitter. Printing Embedded Metric Format JSON to stdout
lets CloudWatch auto-extract metrics with no synchronous API call."""
import json
import time


def emit(score: float, latency_ms: float, decision: str) -> None:
    """Emit one EMF record: FraudScore + Latency, dimensioned by Decision."""
    payload = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [{
                "Namespace": "FraudDetection",
                "Dimensions": [["Decision"]],
                "Metrics": [
                    {"Name": "FraudScore"},
                    {"Name": "LatencyMs", "Unit": "Milliseconds"},
                ],
            }],
        },
        "Decision": decision,
        "FraudScore": score,
        "LatencyMs": latency_ms,
    }
    print(json.dumps(payload))
