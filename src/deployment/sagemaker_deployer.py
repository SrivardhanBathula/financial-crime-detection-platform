import boto3
import sagemaker
from sagemaker.sklearn import SKLearnModel
from sagemaker.xgboost import XGBoostModel
import logging

logger = logging.getLogger(__name__)


class SageMakerDeployer:
    def __init__(self, role: str, region: str = "us-east-1"):
        self.role = role
        self.session = sagemaker.Session()
        self.sm_client = boto3.client("sagemaker", region_name=region)

    def deploy_xgboost(
        self,
        model_data: str,
        endpoint_name: str,
        instance_type: str = "ml.m5.xlarge",
        instance_count: int = 2,
        framework_version: str = "1.7-1"
    ) -> str:
        model = XGBoostModel(
            model_data=model_data,
            role=self.role,
            framework_version=framework_version,
            sagemaker_session=self.session,
        )
        predictor = model.deploy(
            initial_instance_count=instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            data_capture_config=sagemaker.model_monitor.DataCaptureConfig(
                enable_capture=True,
                sampling_percentage=20,
                destination_s3_uri=f"s3://fraud-detection-monitoring/{endpoint_name}"
            )
        )
        logger.info(f"Deployed to endpoint: {endpoint_name}")
        return endpoint_name

    def update_endpoint(self, endpoint_name: str, new_model_data: str):
        config_name = f"{endpoint_name}-config-v2"
        self.sm_client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[{
                "VariantName": "primary",
                "ModelName": endpoint_name,
                "InitialInstanceCount": 2,
                "InstanceType": "ml.m5.xlarge",
                "InitialVariantWeight": 1.0
            }]
        )
        self.sm_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        logger.info(f"Endpoint {endpoint_name} updated with new model")
