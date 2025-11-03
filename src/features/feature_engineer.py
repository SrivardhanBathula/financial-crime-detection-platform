import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)


class TransactionFeatureEngineer:
    """Feature engineering for financial transaction fraud detection."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}
        self.fitted = False

    def _time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "transaction_time" in df.columns:
            dt = pd.to_datetime(df["transaction_time"])
            df["hour"] = dt.dt.hour
            df["day_of_week"] = dt.dt.dayofweek
            df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
            df["is_night"] = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)
        return df

    def _velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "account_id" in df.columns and "amount" in df.columns:
            df["txn_count_1h"] = df.groupby("account_id")["amount"].transform("count")
            df["txn_amount_sum_1h"] = df.groupby("account_id")["amount"].transform("sum")
            df["txn_amount_mean"] = df.groupby("account_id")["amount"].transform("mean")
            df["amount_vs_mean_ratio"] = df["amount"] / (df["txn_amount_mean"] + 1e-6)
        return df

    def _amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "amount" in df.columns:
            df["log_amount"] = np.log1p(df["amount"])
            df["amount_rounded"] = (df["amount"] % 1 == 0).astype(int)
            df["is_large_txn"] = (df["amount"] > df["amount"].quantile(0.95)).astype(int)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._time_features(df)
        df = self._velocity_features(df)
        df = self._amount_features(df)
        self.fitted = True
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Call fit_transform first")
        df = self._time_features(df)
        df = self._velocity_features(df)
        df = self._amount_features(df)
        return df
