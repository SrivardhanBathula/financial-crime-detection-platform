import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """Optimize classification threshold to balance precision/recall for fraud detection."""

    def __init__(self, metric: str = "f1", beta: float = 1.0):
        self.metric = metric
        self.beta = beta
        self.optimal_threshold = 0.5

    def fit(self, y_true, y_proba, thresholds=None) -> float:
        if thresholds is None:
            thresholds = np.arange(0.1, 0.9, 0.01)

        best_score = 0.0
        best_thresh = 0.5

        for thresh in thresholds:
            preds = (y_proba >= thresh).astype(int)
            if self.metric == "f1":
                score = f1_score(y_true, preds, zero_division=0)
            elif self.metric == "f_beta":
                p = precision_score(y_true, preds, zero_division=0)
                r = recall_score(y_true, preds, zero_division=0)
                if p + r > 0:
                    score = (1 + self.beta**2) * p * r / (self.beta**2 * p + r)
                else:
                    score = 0.0
            else:
                score = f1_score(y_true, preds, zero_division=0)

            if score > best_score:
                best_score = score
                best_thresh = thresh

        self.optimal_threshold = best_thresh
        logger.info(f"Optimal threshold: {best_thresh:.3f}, Best {self.metric}: {best_score:.4f}")
        return best_thresh

    def predict(self, y_proba) -> np.ndarray:
        return (y_proba >= self.optimal_threshold).astype(int)
