# System Architecture

## Overview

The Financial Crime Detection Platform processes 500M+ daily financial records
in real-time using a multi-layered AI approach.

## Components

### Layer 1: Data Ingestion
- Apache Kafka for real-time transaction streams
- PySpark Structured Streaming for batch/stream processing
- Databricks Delta Lake for feature storage

### Layer 2: ML Detection
- XGBoost + LightGBM ensemble (F1: 0.87)
- Anomaly detection for novel fraud patterns
- Real-time feature engineering (<5ms latency)

### Layer 3: LLM Investigation
- LangGraph multi-agent workflow
- RAG over Amazon OpenSearch for precedent retrieval
- GPT-4o for automated SAR draft generation

### Layer 4: MLOps
- MLflow for experiment tracking and model registry
- Evidently AI for drift detection
- Prometheus + Grafana for observability
- AWS SageMaker for model serving (99.9% uptime SLA)

## Performance

| Metric | Value |
|--------|-------|
| F1 Score | 0.87 |
| False Positive Reduction | 15% |
| Annual Cost Savings | $2M |
| Analyst Throughput | +30% (40→52 cases/day) |
| System Uptime | 99.9% |
| Processing Latency | <35ms p99 |
