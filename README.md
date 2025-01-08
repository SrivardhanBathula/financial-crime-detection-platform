# 🔍 Intelligent Financial Crime Detection & Risk Analytics Platform

> **Production-grade AI system for real-time fraud detection, risk scoring, and automated financial crime investigation — processing 500M+ financial records with 18% improvement in fraud detection accuracy.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![AWS](https://img.shields.io/badge/AWS-SageMaker-yellow?logo=amazonaws)](https://aws.amazon.com/sagemaker/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Overview

This platform was architected and deployed at **State Street** to detect financial crimes, score transaction risk, and automate investigative workflows using a combination of classical ML, deep learning, and LLM-powered agentic AI.

### 🏆 Key Results
| Metric | Improvement |
|---|---|
| Fraud Detection Accuracy | **+18%** |
| Data Processing Time | **-35%** |
| False Positive Rate | **-15%** |
| Analyst Productivity | **+30%** |
| System Availability | **99.9%** |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                      │
│         Apache Kafka  │  Apache NiFi  │  AWS S3             │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                 PROCESSING & FEATURE LAYER                   │
│         PySpark / Databricks  │  SQL  │  Feature Store       │
└───────────────────────────┬─────────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
┌────────▼───────┐ ┌────────▼───────┐ ┌───────▼────────┐
│  Fraud         │ │  Risk Scoring  │ │  Anomaly       │
│  Detection     │ │  Engine        │ │  Detection     │
│  XGBoost/TF    │ │  LightGBM      │ │  Isolation     │
│                │ │                │ │  Forest/AE     │
└────────┬───────┘ └────────┬───────┘ └───────┬────────┘
         └──────────────────┼──────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│              LLM-POWERED INVESTIGATION LAYER                 │
│         LangChain │ LangGraph │ OpenAI │ RAG (OpenSearch)    │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                  SERVING & OBSERVABILITY                     │
│    FastAPI │ Docker │ Kubernetes │ MLflow │ Grafana          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧠 Core Components

### 1. Real-Time Data Pipeline (`src/data_pipeline/`)
- Kafka-based streaming ingestion of financial transactions
- PySpark transformations on Databricks for 500M+ records
- Feature engineering: velocity features, graph-based entity linking, temporal patterns

### 2. Fraud & Anomaly Detection Models (`src/models/`)
- **XGBoost** ensemble with SHAP explainability for transaction fraud
- **TensorFlow Autoencoder** for unsupervised anomaly detection
- **Graph Neural Network (GNN)** for entity relationship-based fraud rings
- Automated retraining pipelines via Apache Airflow + MLflow

### 3. Risk Scoring Engine (`src/models/risk_scoring.py`)
- Real-time credit and market risk scoring
- Multi-factor risk aggregation with regulatory compliance (AML/BSA)
- Threshold tuning to minimize false positives by 15%

### 4. LLM Investigation Agent (`src/agents/`)
- LangGraph-powered multi-agent workflows for case review automation
- RAG system over Amazon OpenSearch for intelligent case retrieval
- OpenAI GPT-4 integration for natural language case summarization
- Automated SAR (Suspicious Activity Report) draft generation

### 5. Production API (`src/api/`)
- FastAPI microservices with async inference endpoints
- Batch + real-time scoring support
- JWT authentication and rate limiting

### 6. Monitoring & Observability (`src/monitoring/`)
- Prometheus + Grafana dashboards for model and system health
- Evidently AI for real-time data/model drift detection
- CloudWatch alerting for SLA breaches

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **ML/DL Frameworks** | PyTorch, TensorFlow, XGBoost, LightGBM, Scikit-Learn |
| **LLM & Agents** | LangChain, LangGraph, OpenAI API, Amazon OpenSearch (RAG) |
| **Data Engineering** | PySpark, Databricks, Apache Kafka, Apache Airflow, SQL |
| **MLOps** | MLflow, AWS SageMaker, Docker, Kubernetes, CI/CD |
| **Serving** | FastAPI, RESTful Microservices, Async Inference |
| **Observability** | Prometheus, Grafana, Evidently AI, CloudWatch |
| **Cloud** | AWS (S3, SageMaker, OpenSearch, Lambda, Redshift) |

---

## 📁 Project Structure

```
financial-crime-detection-platform/
├── src/
│   ├── data_pipeline/
│   │   ├── kafka_consumer.py          # Real-time transaction ingestion
│   │   ├── spark_feature_engineering.py  # PySpark feature transforms
│   │   └── feature_store.py           # Feature store integration
│   ├── models/
│   │   ├── fraud_detector.py          # XGBoost + TF fraud models
│   │   ├── anomaly_detector.py        # Autoencoder anomaly detection
│   │   ├── risk_scoring.py            # Risk scoring engine
│   │   └── gnn_fraud_rings.py         # GNN for fraud ring detection
│   ├── agents/
│   │   ├── investigation_agent.py     # LangGraph investigation workflow
│   │   ├── rag_retriever.py           # OpenSearch RAG pipeline
│   │   └── sar_generator.py           # Automated SAR report generation
│   ├── api/
│   │   ├── main.py                    # FastAPI application entry
│   │   ├── routers/                   # API route handlers
│   │   └── schemas.py                 # Pydantic request/response models
│   └── monitoring/
│       ├── drift_detector.py          # Evidently AI drift monitoring
│       └── metrics_exporter.py        # Prometheus metrics
├── tests/
│   ├── test_fraud_detector.py
│   ├── test_risk_scoring.py
│   └── test_api.py
├── configs/
│   ├── model_config.yaml
│   └── pipeline_config.yaml
├── notebooks/
│   ├── 01_EDA_Financial_Transactions.ipynb
│   ├── 02_Fraud_Model_Training.ipynb
│   └── 03_LLM_Agent_Demo.ipynb
├── docs/
│   └── architecture.md
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/SrivardhanBathula/financial-crime-detection-platform.git
cd financial-crime-detection-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp configs/.env.example configs/.env
# Edit configs/.env with your API keys and AWS credentials

# Run the API locally
uvicorn src.api.main:app --reload --port 8000

# Run with Docker
docker-compose up --build
```

---

## 📊 Model Performance

| Model | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|
| XGBoost Fraud Detector | 0.94 | 0.91 | 0.92 | 0.97 |
| TF Autoencoder (Anomaly) | 0.89 | 0.87 | 0.88 | 0.95 |
| GNN Fraud Ring Detector | 0.91 | 0.89 | 0.90 | 0.96 |
| Risk Scoring Ensemble | 0.93 | 0.90 | 0.91 | 0.96 |

---

## 🔒 Compliance & Security
- AML/BSA regulatory compliance built into scoring logic
- GDPR and CCPA-compliant data handling
- All PII masked in logs and monitoring dashboards
- Role-based access control on all API endpoints

---

## 👤 Author

**Srivardhan Bathula** — AI/ML Engineer  
📧 Srivardhan.Bathula1@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/srivardhan-bathula) | [GitHub](https://github.com/SrivardhanBathula)
