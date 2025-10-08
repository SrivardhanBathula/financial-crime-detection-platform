import os
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

@dataclass
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    aws_region: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    sagemaker_role: str = os.getenv("SAGEMAKER_ROLE", "")
    kafka_bootstrap: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    mlflow_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()
