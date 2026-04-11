from prometheus_client import Counter, Histogram, Gauge, Summary
from functools import wraps
import time
import logging

logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTIONS_TOTAL = Counter("fraud_predictions_total", "Total predictions", ["model_version", "result"])
PREDICTION_LATENCY = Histogram("fraud_prediction_latency_seconds", "Prediction latency",
                                ["model_version"], buckets=[.005, .01, .025, .05, .1, .25, .5, 1.0])
FALSE_POSITIVE_RATE = Gauge("fraud_false_positive_rate", "Current false positive rate")
MODEL_F1_SCORE = Gauge("fraud_model_f1_score", "Current model F1 score", ["model_version"])
ACTIVE_INVESTIGATIONS = Gauge("fraud_active_investigations", "Active LangGraph investigations")
KAFKA_RECORDS_PROCESSED = Counter("fraud_kafka_records_total", "Kafka records processed", ["topic"])
DRIFT_SCORE = Gauge("fraud_drift_score", "Current data drift score", ["feature_group"])


def track_prediction(model_version: str = "v3"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                latency = time.time() - start
                label = "fraud" if result.get("is_fraud") else "legitimate"
                PREDICTIONS_TOTAL.labels(model_version=model_version, result=label).inc()
                PREDICTION_LATENCY.labels(model_version=model_version).observe(latency)
                return result
            except Exception as e:
                PREDICTIONS_TOTAL.labels(model_version=model_version, result="error").inc()
                raise
        return wrapper
    return decorator
