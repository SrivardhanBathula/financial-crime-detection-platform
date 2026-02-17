from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType
from pyspark.ml.feature import VectorAssembler, StandardScaler
import logging

logger = logging.getLogger(__name__)


class FraudDetectionSparkPipeline:
    def __init__(self, app_name: str = "FraudDetection"):
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.shuffle.partitions", "200") \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("WARN")

    def read_kafka_stream(self, bootstrap_servers: str, topic: str) -> DataFrame:
        return self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", bootstrap_servers) \
            .option("subscribe", topic) \
            .option("startingOffsets", "latest") \
            .option("maxOffsetsPerTrigger", 100000) \
            .load()

    def transform_transactions(self, df: DataFrame) -> DataFrame:
        from pyspark.sql.functions import from_json, col, hour, dayofweek, log1p
        from pyspark.sql.types import StructType, StructField, DoubleType, StringType, TimestampType
        schema = StructType([
            StructField("transaction_id", StringType()),
            StructField("account_id", StringType()),
            StructField("amount", DoubleType()),
            StructField("merchant_id", StringType()),
            StructField("transaction_time", TimestampType()),
        ])
        return df \
            .select(from_json(col("value").cast("string"), schema).alias("data")) \
            .select("data.*") \
            .withColumn("log_amount", log1p(col("amount"))) \
            .withColumn("hour", hour(col("transaction_time"))) \
            .withColumn("day_of_week", dayofweek(col("transaction_time"))) \
            .withColumn("is_weekend", (col("day_of_week") >= 6).cast(IntegerType())) \
            .withColumn("is_night", ((col("hour") < 6) | (col("hour") > 22)).cast(IntegerType()))

    def compute_velocity(self, df: DataFrame, window_duration: str = "1 hour") -> DataFrame:
        from pyspark.sql.functions import window, count, sum as spark_sum, avg
        return df.groupBy(
            F.window("transaction_time", window_duration),
            "account_id"
        ).agg(
            count("*").alias("txn_count"),
            spark_sum("amount").alias("total_amount"),
            avg("amount").alias("avg_amount")
        )
