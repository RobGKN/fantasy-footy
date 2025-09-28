# src/distill/cart.py
"""
CART surrogate with two modes:
- regression (DecisionTreeRegressor on teacher logits)
- classification (DecisionTreeClassifier for shortlist membership)

Public API:
- CartStudent.fit(X, targets, cfg, sample_weight=None)
  * cfg['mode'] in {'regression','classification'}
- CartStudent.predict_logits(X)  # returns logits; sigmoid(logits)=proba
- CartStudent.export_rules(feature_names, max_depth=None)
- CartStudent.save(out_dir) / load(out_dir)
"""

from __future__ import annotations

from typing import Dict, List, Optional
import os, json, logging
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text
from sklearn import __version__ as sklearn_version
import joblib


def _logit(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


class CartStudent:
    def __init__(self) -> None:
        self.mode: str = "regression"
        self.tree: Optional[object] = None
        self.meta: Dict = {}

    def fit(
        self,
        X: np.ndarray,
        targets: np.ndarray,
        cfg: Dict,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "CartStudent":
        if X.ndim != 2:
            raise ValueError(f"X must be 2D [N,F], got {X.shape}")
        N = X.shape[0]
        targets = np.asarray(targets)
        if targets.shape[0] != N:
            raise ValueError("targets length must match X.shape[0]")
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1)
            if sample_weight.shape[0] != N:
                raise ValueError("sample_weight length must match X.shape[0]")

        self.mode = cfg.get("mode", "regression")
        max_depth = int(cfg.get("max_depth", 5))
        min_samples_leaf = int(cfg.get("min_samples_leaf", 50))
        ccp_alpha = float(cfg.get("ccp_alpha", 0.0))
        seed = int(cfg.get("seed", 42))

        logging.info(
            f"Fitting CART ({self.mode}): max_depth={max_depth}, "
            f"min_samples_leaf={min_samples_leaf}, ccp_alpha={ccp_alpha}, seed={seed}"
            + (", weighted" if sample_weight is not None else "")
        )

        if self.mode == "classification":
            y = targets.astype(int).reshape(-1)
            tree = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                ccp_alpha=ccp_alpha,
                random_state=seed,
            )
            tree.fit(X, y, sample_weight=sample_weight)
        else:
            z = targets.astype(float).reshape(-1)
            tree = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                ccp_alpha=ccp_alpha,
                random_state=seed,
            )
            tree.fit(X, z, sample_weight=sample_weight)

        self.tree = tree
        self.meta = {
            "mode": self.mode,
            "sklearn_version": sklearn_version,
            "params": tree.get_params(),
            "structure": {
                "n_nodes": int(tree.tree_.node_count),
                "max_depth": int(tree.tree_.max_depth),
            },
        }
        return self

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        if self.tree is None:
            raise RuntimeError("CartStudent is not fitted.")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D [N,F], got {X.shape}")
        if self.mode == "classification":
            proba = self.tree.predict_proba(X)[:, 1]  # P(shortlist=1)
            return _logit(proba)
        else:
            pred_z = self.tree.predict(X)
            return np.asarray(pred_z, dtype=float)

    def export_rules(self, feature_names: List[str], max_depth: Optional[int] = None) -> str:
        if self.tree is None:
            raise RuntimeError("CartStudent is not fitted.")
        if max_depth is None:
            try:
                max_depth = int(self.tree.get_depth())
            except Exception:
                max_depth = int(getattr(self.tree, "tree_", None).max_depth)
        return export_text(self.tree, feature_names=feature_names, max_depth=max_depth)

    def save(self, out_dir: str) -> None:
        if self.tree is None:
            raise RuntimeError("Cannot save an unfitted CartStudent.")
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(self.tree, os.path.join(out_dir, "tree.pkl"))
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2)
        logging.info(f"Saved CART surrogate to {os.path.join(out_dir,'tree.pkl')} and metadata to {os.path.join(out_dir,'meta.json')}")

    @classmethod
    def load(cls, out_dir: str) -> "CartStudent":
        obj = cls()
        obj.tree = joblib.load(os.path.join(out_dir, "tree.pkl"))
        with open(os.path.join(out_dir, "meta.json"), "r") as f:
            obj.meta = json.load(f)
        obj.mode = obj.meta.get("mode", "regression")
        return obj
