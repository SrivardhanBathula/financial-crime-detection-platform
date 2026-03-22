import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import f1_score
import mlflow
import logging

logger = logging.getLogger(__name__)


class FraudEnsembleModel:
    """Stacking ensemble: XGBoost + LightGBM + Logistic Regression meta-learner."""

    def __init__(self):
        self.xgb = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05,
                                       scale_pos_weight=10, tree_method="hist",
                                       eval_metric="aucpr", use_label_encoder=False)
        self.lgb = lgb.LGBMClassifier(n_estimators=500, max_depth=8, learning_rate=0.05,
                                        class_weight="balanced", verbose=-1)
        self.meta = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
        self.stacker = StackingClassifier(
            estimators=[("xgb", self.xgb), ("lgb", self.lgb)],
            final_estimator=self.meta,
            cv=5, passthrough=True, n_jobs=-1
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        with mlflow.start_run(run_name="ensemble_v3"):
            self.stacker.fit(X_train, y_train)
            if X_val is not None:
                preds = self.stacker.predict(X_val)
                f1 = f1_score(y_val, preds)
                mlflow.log_metric("val_f1", f1)
                mlflow.log_param("ensemble_type", "stacking_xgb_lgb_lr")
                logger.info(f"Ensemble F1: {f1:.4f}")
            mlflow.sklearn.log_model(self.stacker, "ensemble_model",
                                    registered_model_name="fraud_ensemble_v3")
        return self

    def predict_proba(self, X):
        return self.stacker.predict_proba(X)[:, 1]
