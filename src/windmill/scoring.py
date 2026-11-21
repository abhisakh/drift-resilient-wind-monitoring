"""CARE score: Coverage, Accuracy, Reliability, Earliness.

The definition here follows the CARE to Compare paper (Gück, Roelofs & Faulstich,
arXiv:2404.10320 / MDPI Data 9(12):138, 2024) exactly:

  Coverage    pointwise F-score with beta = 1/2 on anomaly datasets, computed over
              the data points whose status-ID is "normal operation".
  Accuracy    tn / (tn + fp) on normal datasets (datasets with no anomaly at all).
  Reliability event-wise F-score with beta = 1/2 across all datasets, where an
              event counts as detected when the maximum *criticality* reaches the
              threshold (default 72 = 12 h of consecutive 10-min detections).
  Earliness   weighted score: detections in the first quarter of the event carry
              full weight, decaying linearly to zero at the event end.

The final score is a weighted average with weights coverage=1, earliness=1,
reliability=1, accuracy=2 (the accuracy term is doubled so that the normal
datasets count as much as the anomaly ones), with two special cases: no anomaly
detected at all gives 0, and average accuracy below 0.5 gives that accuracy.

This module is a self-contained re-implementation so the project does not depend on
the authors' package at runtime. `scripts/04_score.py` verifies it against the
authors' reference implementation (`energy-fault-detector` 0.8.1) event by event;
`tests/test_scoring.py` pins the behaviour.

Published baselines to beat: AE 0.66 | random 0.5 | iForest < 0.5 | trivial 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, fbeta_score

# The CARE paper's weighting of the four sub-scores.
COVERAGE_W = 1.0
ACCURACY_W = 2.0
RELIABILITY_W = 1.0
EARLINESS_W = 1.0

COVERAGE_BETA = 0.5
RELIABILITY_BETA = 0.5

#: Criticality threshold from the paper: 72 consecutive 10-min detections = 12 h.
CRITICALITY_THRESHOLD = 72

#: Fraction of the event over which the earliness weight is flat (paper default).
WS_START_OF_DESCEND = (1, 4)

#: Status-IDs the dataset considers "normal operation" (paper table 3.3).
NORMAL_STATUS_IDS = (0, 2)


def calculate_criticality(
    anomalies: pd.Series, normal_idx: pd.Series | None = None
) -> pd.Series:
    """Algorithm 1 of the CARE paper.

    Criticality increases by one for each detected anomaly during normal
    operation, decreases by one (floor 0) for each normal reading during normal
    operation, and is held constant while the reading is not normal operation.

    Returns a Series indexed like `anomalies`.
    """
    anomalies = anomalies.sort_index()
    if normal_idx is None:
        normal_idx = pd.Series(True, index=anomalies.index)
    else:
        normal_idx = normal_idx.sort_index().reindex(anomalies.index)
        normal_idx = normal_idx.fillna(True).astype(bool)

    if len(anomalies) != len(normal_idx):
        raise ValueError("length mismatch between anomalies and normal_idx")

    a = anomalies.to_numpy(dtype=bool)
    n = normal_idx.to_numpy(dtype=bool)
    crit = np.zeros(len(a), dtype="int64")
    cur = 0
    for i in range(len(a)):
        if not n[i]:
            cur = cur            # held constant while not in normal operation
        elif a[i]:
            cur = cur + 1
        else:
            cur = max(cur - 1, 0)
        crit[i] = cur
    return pd.Series(crit, index=anomalies.index)


def weighted_score(event_prediction: pd.Series) -> float:
    """Earliness: weighted score of one event's predictions.

    The weight is 1 over the first quarter of the event and decays linearly to 0
    at the event end, so early detections count for more.
    """
    event_prediction = event_prediction.sort_index()
    length = len(event_prediction)
    if length == 0:
        return float("nan")

    numerator, denominator = WS_START_OF_DESCEND
    start_of_descend = numerator / denominator
    scale = int(denominator)
    scaled_length = length * scale
    cp = int(scaled_length * start_of_descend)

    x_values = np.linspace(0, scale, scaled_length)
    weights = np.zeros(scaled_length)
    weights[:cp] = scale
    slope = 1 / (1 - start_of_descend)
    offset = scale / (1 - start_of_descend)
    weights[cp:scaled_length] = offset - slope * x_values[cp:scaled_length]

    final_weights = weights[::scale]
    weights = final_weights / scale

    p = event_prediction.to_numpy(dtype=bool)
    denom = weights.sum()
    if denom == 0:
        return float("nan")
    return float((p * weights).sum() / denom)


def create_ground_truth(
    event_start, event_end, normal_index: pd.Series, event_label: str
) -> pd.Series:
    """Ground truth: True where behaviour is not normal, plus the event window."""
    gt = pd.Series(data=~normal_index, index=normal_index.index, name="ground_truth")
    gt = gt.sort_index()
    if event_label == "anomaly":
        gt.loc[event_start:event_end] = True
    return gt


class CAREScore:
    """The CARE score, event by event."""

    def __init__(
        self,
        coverage_beta: float = COVERAGE_BETA,
        reliability_beta: float = RELIABILITY_BETA,
        coverage_w: float = COVERAGE_W,
        accuracy_w: float = ACCURACY_W,
        earliness_w: float = EARLINESS_W,
        reliability_w: float = RELIABILITY_W,
        criticality_threshold: int = CRITICALITY_THRESHOLD,
    ):
        self.coverage_beta = coverage_beta
        self.reliability_beta = reliability_beta
        self.coverage_w = coverage_w
        self.accuracy_w = accuracy_w
        self.earliness_w = earliness_w
        self.reliability_w = reliability_w
        self.criticality_threshold = criticality_threshold
        self._events: list[dict] = []

    @property
    def evaluated_events(self) -> pd.DataFrame:
        df = pd.DataFrame(self._events)
        if not df.empty:
            df["anomaly_detected"] = df["max_criticality"] >= self.criticality_threshold
        return df

    def evaluate_event(
        self,
        event_start,
        event_end,
        event_label: str,
        predicted_anomalies: pd.Series,
        normal_index: pd.Series | None = None,
        event_id: int | None = None,
    ) -> dict:
        """Evaluate one event.

        `predicted_anomalies` is a boolean Series over the dataset's rows;
        `normal_index` marks rows in normal operation (status-ID 0 or 2).
        """
        if event_label not in ("anomaly", "normal"):
            raise ValueError("event_label must be 'anomaly' or 'normal'")

        predicted_anomalies = predicted_anomalies.sort_index()
        if normal_index is None:
            normal_index = pd.Series(True, index=predicted_anomalies.index)
        else:
            normal_index = normal_index.sort_index()

        ground_truth = create_ground_truth(event_start, event_end, normal_index, event_label)
        event_pred = predicted_anomalies.loc[event_start:event_end]
        if event_pred.empty:
            raise ValueError("no event data could be selected")

        weighted = weighted_score(event_pred) if event_label == "anomaly" else np.nan

        criticality = calculate_criticality(
            predicted_anomalies.loc[:event_end], normal_index.loc[:event_end]
        )
        max_criticality = int(np.max(criticality)) if len(criticality) else 0

        # Metrics are only computed on rows in normal operation.
        gt_n = ground_truth[normal_index]
        p_n = predicted_anomalies[normal_index]
        if event_label == "anomaly":
            f_score = fbeta_score(y_true=gt_n, y_pred=p_n, beta=self.coverage_beta,
                                   zero_division=0)
        else:
            f_score = np.nan
        acc = accuracy_score(y_true=gt_n, y_pred=p_n)
        tn, fp, fn, tp = (int(v) for v in
                           pd.crosstab(gt_n, p_n).reindex(index=[False, True],
                                                          columns=[False, True],
                                                          fill_value=0).to_numpy().ravel())

        evaluation = {
            "event_id": event_id if event_id is not None else len(self._events),
            "event_label": event_label,
            "weighted_score": weighted,
            "max_criticality": max_criticality,
            "f_beta_score": f_score,
            "accuracy": acc,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        }
        self._events.append(evaluation)
        return evaluation

    def calculate_reliability(self) -> float:
        """Event-wise F-score across all evaluated events."""
        df = self.evaluated_events
        if df.empty:
            return float("nan")
        return float(fbeta_score(
            y_true=(df["event_label"] == "anomaly").to_numpy(),
            y_pred=df["anomaly_detected"].to_numpy(),
            beta=self.reliability_beta, zero_division=0))

    def get_final_score(self) -> float:
        """The CARE score over all evaluated events."""
        df = self.evaluated_events
        if df.empty:
            return 0.0
        is_anomaly = df["event_label"] == "anomaly"
        if not is_anomaly.any() or is_anomaly.all():
            raise ValueError("need at least one anomaly event and one normal event")

        avg_accuracy = float(df.loc[~is_anomaly, "accuracy"].mean())
        if avg_accuracy <= 0.5:
            return avg_accuracy
        if df["anomaly_detected"].sum() == 0:
            return 0.0

        avg_f = float(df.loc[is_anomaly, "f_beta_score"].mean())
        avg_ws = float(df.loc[is_anomaly, "weighted_score"].mean())
        reliability = self.calculate_reliability()

        total = (avg_accuracy * self.accuracy_w
                 + avg_ws * self.earliness_w
                 + avg_f * self.coverage_w
                 + reliability * self.reliability_w)
        weights = (self.accuracy_w + self.earliness_w
                   + self.coverage_w + self.reliability_w)
        return float(total / weights)

    def sub_scores(self) -> dict:
        """The four sub-scores, for reporting."""
        df = self.evaluated_events
        is_anomaly = df["event_label"] == "anomaly"
        return {
            "coverage": round(float(df.loc[is_anomaly, "f_beta_score"].mean()), 4),
            "accuracy": round(float(df.loc[~is_anomaly, "accuracy"].mean()), 4),
            "reliability": round(self.calculate_reliability(), 4),
            "earliness": round(float(df.loc[is_anomaly, "weighted_score"].mean()), 4),
        }
