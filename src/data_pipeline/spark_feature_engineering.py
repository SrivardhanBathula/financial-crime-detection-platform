"""
PySpark Feature Engineering for Financial Crime Detection
Processes 500M+ transaction records on Databricks for fraud/risk modeling.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, IntegerType, StringType


def get_spark_session(app_name: str = "FinancialCrimeFeatureEngineering") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )


def load_transactions(spark: SparkSession, path: str) -> DataFrame:
    """Load raw transaction data from S3 / Delta Lake."""
    return spark.read.format("delta").option("mergeSchema", "true").load(path)


def engineer_velocity_features(df: DataFrame) -> DataFrame:
    """
    Compute velocity-based features:
    - Transaction counts and amounts in 1h, 24h, 7d windows
    - Merchant and location frequency per account
    """
    # Window specs
    w_1h = Window.partitionBy("account_id").orderBy(F.col("timestamp").cast("long")).rangeBetween(-3600, 0)
    w_24h = Window.partitionBy("account_id").orderBy(F.col("timestamp").cast("long")).rangeBetween(-86400, 0)
    w_7d = Window.partitionBy("account_id").orderBy(F.col("timestamp").cast("long")).rangeBetween(-604800, 0)

    return df.withColumn("txn_count_1h", F.count("transaction_id").over(w_1h)) \
             .withColumn("txn_count_24h", F.count("transaction_id").over(w_24h)) \
             .withColumn("txn_count_7d", F.count("transaction_id").over(w_7d)) \
             .withColumn("txn_amount_sum_1h", F.sum("amount").over(w_1h)) \
             .withColumn("txn_amount_sum_24h", F.sum("amount").over(w_24h)) \
             .withColumn("txn_amount_avg_7d", F.avg("amount").over(w_7d)) \
             .withColumn("txn_amount_stddev_7d", F.stddev("amount").over(w_7d))


def engineer_merchant_features(df: DataFrame) -> DataFrame:
    """
    Merchant-level aggregations:
    - First-time merchant flag
    - Merchant risk score (pre-computed lookup)
    - Cross-border transaction flag
    """
    account_merchant_window = Window.partitionBy("account_id", "merchant_id").orderBy("timestamp")

    return df.withColumn("merchant_rank", F.rank().over(account_merchant_window)) \
             .withColumn("is_first_merchant_visit", (F.col("merchant_rank") == 1).cast(IntegerType())) \
             .withColumn("is_cross_border",
                         (F.col("account_country") != F.col("merchant_country")).cast(IntegerType())) \
             .withColumn("amount_vs_merchant_avg",
                         F.col("amount") / (F.avg("amount").over(
                             Window.partitionBy("merchant_id").rowsBetween(-1000, 0)) + 1e-6))


def engineer_temporal_features(df: DataFrame) -> DataFrame:
    """
    Time-based behavioral features:
    - Hour of day, day of week
    - Is weekend / night transaction
    - Time since last transaction
    """
    account_time_window = Window.partitionBy("account_id").orderBy("timestamp")

    return df.withColumn("hour_of_day", F.hour("timestamp")) \
             .withColumn("day_of_week", F.dayofweek("timestamp")) \
             .withColumn("is_weekend", (F.dayofweek("timestamp").isin([1, 7])).cast(IntegerType())) \
             .withColumn("is_night_txn",
                         ((F.hour("timestamp") < 6) | (F.hour("timestamp") > 22)).cast(IntegerType())) \
             .withColumn("prev_timestamp", F.lag("timestamp").over(account_time_window)) \
             .withColumn("seconds_since_last_txn",
                         (F.col("timestamp").cast("long") - F.col("prev_timestamp").cast("long")))


def engineer_amount_features(df: DataFrame) -> DataFrame:
    """
    Amount-based statistical features:
    - Z-score of transaction amount vs account history
    - Round amount flag (potential structuring indicator)
    - Log-transformed amount
    """
    account_window = Window.partitionBy("account_id")

    return df.withColumn("account_avg_amount", F.avg("amount").over(account_window)) \
             .withColumn("account_stddev_amount", F.stddev("amount").over(account_window)) \
             .withColumn("amount_zscore",
                         (F.col("amount") - F.col("account_avg_amount")) /
                         (F.col("account_stddev_amount") + 1e-6)) \
             .withColumn("is_round_amount",
                         ((F.col("amount") % 100 == 0) | (F.col("amount") % 1000 == 0)).cast(IntegerType())) \
             .withColumn("log_amount", F.log1p(F.col("amount").cast(DoubleType())))


def build_feature_set(spark: SparkSession, input_path: str, output_path: str):
    """
    Full feature engineering pipeline: load → transform → write to feature store.
    """
    print("Loading raw transactions...")
    df = load_transactions(spark, input_path)

    print("Engineering velocity features...")
    df = engineer_velocity_features(df)

    print("Engineering merchant features...")
    df = engineer_merchant_features(df)

    print("Engineering temporal features...")
    df = engineer_temporal_features(df)

    print("Engineering amount features...")
    df = engineer_amount_features(df)

    # Drop intermediate columns
    df = df.drop("prev_timestamp", "merchant_rank", "account_avg_amount", "account_stddev_amount")

    print(f"Writing {df.count():,} enriched records to feature store...")
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(output_path)
    print("Feature engineering complete.")


if __name__ == "__main__":
    spark = get_spark_session()
    build_feature_set(
        spark,
        input_path="s3://your-bucket/raw/transactions/",
        output_path="s3://your-bucket/features/transactions_enriched/",
    )
