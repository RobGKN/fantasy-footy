from typing import Dict, List
import numpy as np

class CartStudent:
    """
    Axis-aligned CART surrogate trained on teacher logits.
    API exposes predict_logits(X) to plug into existing evaluation.
    """
    def __init__(self):
        self.tree = None
        self.meta = {}

    def fit(self, X: np.ndarray, teacher_logits: np.ndarray, cfg: Dict) -> "CartStudent":
        """Fit a DecisionTreeRegressor with max_depth/min_samples_leaf/ccp_alpha from cfg."""
        raise NotImplementedError

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        """Return predicted logits [N]."""
        raise NotImplementedError

    def export_rules(self, feature_names: List[str], max_depth: int = None) -> str:
        """Return a plain-text rules dump (sklearn export_text)."""
        raise NotImplementedError

    def save(self, out_dir: str) -> None:
        """Persist tree.pkl and meta.json to out_dir."""
        raise NotImplementedError

    @classmethod
    def load(cls, out_dir: str) -> "CartStudent":
        """Load tree and meta from out_dir."""
        raise NotImplementedError