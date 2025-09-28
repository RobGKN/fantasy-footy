# src/distill/metrics.py
"""
Metrics for evaluating a CART surrogate distilled from the MLP teacher.

Provided functions:
- spearman_fidelity(teacher_scores, student_scores) -> float
- topk_overlap(teacher_scores, student_scores, k) -> float
- evaluate_team_value_with_existing_pipeline(probs, labels, val_df, k, random_trials) -> Dict

Notes:
- We avoid adding new dependencies by implementing a light Spearman correlation
  via rank + Pearson. For typical continuous logits there are no ties; if ties
  occur, ranks are still well-defined (no tie-averaging).
"""

from typing import Dict
import numpy as np
import pandas as pd

# Reuse your existing evaluation helpers for AUC and team-value:
from src.evaluation import evaluate_team_value, evaluate_roc_auc


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Numerically stable Pearson correlation on 1D arrays."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError(f"Length mismatch: {a.size} vs {b.size}")
    if a.size == 0:
        return np.nan
    a_mean = a.mean()
    b_mean = b.mean()
    a_c = a - a_mean
    b_c = b - b_mean
    denom = np.linalg.norm(a_c) * np.linalg.norm(b_c)
    if denom == 0.0:
        # Degenerate variance on one side → undefined correlation
        return np.nan
    return float(np.dot(a_c, b_c) / denom)


def _rank(data: np.ndarray) -> np.ndarray:
    """
    Simple rank implementation (0..N-1). For ties, this produces a consistent,
    deterministic ordering based on stable argsort — acceptable for our use.
    """
    data = np.asarray(data)
    order = np.argsort(data, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(data), dtype=float)
    return ranks


def spearman_fidelity(teacher_scores: np.ndarray, student_scores: np.ndarray) -> float:
    """
    Spearman rank correlation between teacher and student scores.
    Intended for comparing teacher logits (or probs) vs student logits (or probs).
    """
    r_t = _rank(teacher_scores)
    r_s = _rank(student_scores)
    return _pearson_corr(r_t, r_s)


def topk_overlap(teacher_scores: np.ndarray, student_scores: np.ndarray, k: int) -> float:
    """
    Fractional overlap between top-k by teacher_scores and top-k by student_scores.
    """
    if k <= 0:
        return 0.0
    t_idx = np.argsort(-np.asarray(teacher_scores))[:k]
    s_idx = np.argsort(-np.asarray(student_scores))[:k]
    inter = len(set(t_idx.tolist()).intersection(set(s_idx.tolist())))
    return inter / float(k)


def evaluate_team_value_with_existing_pipeline(
    probs: np.ndarray,
    labels: np.ndarray,
    val_df: pd.DataFrame,
    k: int,
    random_trials: int,
) -> Dict:
    """
    Use your existing evaluation helpers to compute:
      - auc
      - model_team_value
      - random_team_value

    Args:
        probs: Predicted probabilities (after sigmoid) for the validation set, shape [N].
        labels: Ground-truth binary labels aligned with probs, shape [N].
        val_df: Validation DataFrame aligned row-wise with probs/labels and containing
                'future_avg_points' and 'Next_Year_Price' for team-value computation.
        k: Team size for value evaluation (e.g., 22).
        random_trials: Number of random team draws for the baseline.

    Returns:
        dict with keys: 'auc', 'model_team_value', 'random_team_value'
    """
    probs = np.asarray(probs).ravel()
    labels = np.asarray(labels).ravel()
    if probs.size != labels.size:
        raise ValueError(f"Size mismatch: probs={probs.size} vs labels={labels.size}")

    auc = float(evaluate_roc_auc(probs, labels))
    model_team_value, random_team_value = evaluate_team_value(
        probs, labels, val_df.copy(), k=k, random_trials=random_trials
    )
    return {
        "auc": auc,
        "model_team_value": float(model_team_value),
        "random_team_value": float(random_team_value),
    }
