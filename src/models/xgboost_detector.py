import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import mlflow
import mlflow.xgboost
import logging

logger = logging.getLogger(__name__)


class FraudDetectionModel:
    def __init__(self, config: dict = None):
        self.config = config or {
            "n_estimators": 500, "max_depth": 8, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": 10,
            "eval_metric": "aucpr", "use_label_encoder": False, "tree_method": "hist"
        }
        self.model = xgb.XGBClassifier(**self.config)
        self.feature_importance_ = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(X_train, y_train, eval_set=eval_set,
                      verbose=100, early_stopping_rounds=50)
        self.feature_importance_ = pd.Series(
            self.model.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False)
        return self

    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1]
        metrics = {
            "f1": round(f1_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds), 4),
            "recall": round(recall_score(y_test, preds), 4),
            "auc_roc": round(roc_auc_score(y_test, probs), 4),
        }
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
