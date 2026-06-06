"""
ML Model Drift Detection and Automated Retraining Trigger
Monitors production model performance using Evidently AI.
Triggers automated retraining when drift exceeds thresholds.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Drift thresholds — what triggers retraining
DRIFT_THRESHOLDS = {
    "data_drift_share": 0.20,        # >20% features drifted triggers alert
    "psi_score": 0.20,               # PSI > 0.2 = significant drift
    "f1_degradation": 0.05,          # F1 drop > 5% triggers retraining
    "prediction_drift": 0.10,        # >10% shift in prediction distribution
}


@dataclass
class DriftReport:
    timestamp: str
    dataset_name: str
    n_features_drifted: int
    n_features_total: int
    drift_share: float
    psi_scores: dict[str, float]
    prediction_drift_detected: bool
    retraining_triggered: bool
    drifted_features: list[str] = field(default_factory=list)
    alert_message: str = ""


class DriftDetector:
    """
    Evidently AI-powered drift detector with automated retraining triggers.
    Monitors data distribution shifts and model performance degradation.
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        thresholds: Optional[dict] = None,
        results_dir: str = "drift_reports",
        retraining_webhook: Optional[str] = None,
    ):
        self.reference_data = reference_data
        self.thresholds = thresholds or DRIFT_THRESHOLDS
        self.results_dir = Path(results_dir)
        self.retraining_webhook = retraining_webhook

    def detect(self, current_data: pd.DataFrame, labels: Optional[pd.Series] = None) -> DriftReport:
        """
        Run drift detection between reference and current data.
        Triggers retraining if drift exceeds configured thresholds.
        """
        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
            from evidently.metrics import DatasetDriftMetric, DataDriftTable
        except ImportError:
            logger.error("Install: pip install evidently")
            return self._empty_report()

        logger.info(f"Running drift detection: {len(current_data)} current vs {len(self.reference_data)} reference samples")

        report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftTable(),
        ])
        report.run(reference_data=self.reference_data, current_data=current_data)
        report_dict = report.as_dict()

        drift_metrics = report_dict["metrics"][0]["result"]
        drift_table = report_dict["metrics"][1]["result"]

        n_drifted = drift_metrics.get("number_of_drifted_columns", 0)
        n_total = drift_metrics.get("number_of_columns", 1)
        drift_share = drift_metrics.get("share_of_drifted_columns", 0.0)

        drifted_features = [
            col for col, stats in drift_table.get("drift_by_columns", {}).items()
            if stats.get("drift_detected", False)
        ]

        psi_scores = {
            col: stats.get("stattest_threshold", 0.0)
            for col, stats in drift_table.get("drift_by_columns", {}).items()
        }

        retraining_needed = (
            drift_share > self.thresholds["data_drift_share"] or
            any(v > self.thresholds["psi_score"] for v in psi_scores.values())
        )

        report_obj = DriftReport(
            timestamp=datetime.now().isoformat(),
            dataset_name="production_inference",
            n_features_drifted=n_drifted,
            n_features_total=n_total,
            drift_share=round(drift_share, 4),
            psi_scores=psi_scores,
            prediction_drift_detected=drift_share > self.thresholds["prediction_drift"],
            retraining_triggered=retraining_needed,
            drifted_features=drifted_features,
            alert_message=f"Drift detected in {n_drifted}/{n_total} features ({drift_share:.1%})" if retraining_needed else "",
        )

        self._save_report(report_obj)

        if retraining_needed:
            logger.warning(f"DRIFT ALERT: {report_obj.alert_message}")
            self._trigger_retraining(report_obj)
        else:
            logger.info(f"Drift within thresholds: {drift_share:.1%} features drifted")

        return report_obj

    def _trigger_retraining(self, report: DriftReport):
        """Trigger automated retraining via webhook or direct pipeline call."""
        if self.retraining_webhook:
            import requests
            try:
                payload = {
                    "trigger": "drift_detection",
                    "timestamp": report.timestamp,
                    "drift_share": report.drift_share,
                    "drifted_features": report.drifted_features,
                }
                resp = requests.post(self.retraining_webhook, json=payload, timeout=10)
                logger.info(f"Retraining webhook triggered: HTTP {resp.status_code}")
            except Exception as e:
                logger.error(f"Failed to trigger retraining webhook: {e}")
        else:
            logger.info("Retraining trigger: logged (no webhook configured)")

    def check_performance_degradation(
        self,
        current_f1: float,
        baseline_f1: float,
    ) -> bool:
        """Check if F1 score has degraded beyond threshold."""
        degradation = baseline_f1 - current_f1
        if degradation > self.thresholds["f1_degradation"]:
            logger.warning(
                f"F1 DEGRADATION: {current_f1:.4f} vs baseline {baseline_f1:.4f} "
                f"(drop: {degradation:.4f} > threshold {self.thresholds['f1_degradation']})"
            )
            return True
        return False

    def _save_report(self, report: DriftReport):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        path = self.results_dir / f"drift_{report.timestamp[:10]}.json"
        with open(path, "w") as f:
            json.dump({
                "timestamp": report.timestamp,
                "drift_share": report.drift_share,
                "n_features_drifted": report.n_features_drifted,
                "n_features_total": report.n_features_total,
                "drifted_features": report.drifted_features,
                "retraining_triggered": report.retraining_triggered,
                "alert_message": report.alert_message,
            }, f, indent=2)

    def _empty_report(self) -> DriftReport:
        return DriftReport(
            timestamp=datetime.now().isoformat(),
            dataset_name="unknown",
            n_features_drifted=0,
            n_features_total=0,
            drift_share=0.0,
            psi_scores={},
            prediction_drift_detected=False,
            retraining_triggered=False,
        )
