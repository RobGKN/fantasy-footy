from typing import Dict, Tuple, List
import torch
import numpy as np
import pandas as pd

def load_trained_teacher(path: str, device: str = "cpu") -> torch.nn.Module:
    """Load BreakoutPredictorMLP with weights at `path`; return model.eval()."""
    raise NotImplementedError

def build_teacher_dataloaders(config: Dict) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, pd.DataFrame]:
    """
    Construct train/val dataloaders consistent with your training split.
    Return (train_loader, val_loader, val_df_in_order) where val_df aligns row-wise to val_loader.
    """
    raise NotImplementedError

def collect_teacher_logits(
    teacher: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str = "cpu",
    return_labels: bool = True
) -> Dict[str, np.ndarray]:
    """
    Run teacher over `loader` and return dict:
      - 'X': features [N, F]
      - 'logits': raw logits [N]
      - 'y': labels [N] if return_labels else omitted
    """
    raise NotImplementedError

def get_feature_names() -> List[str]:
    """Return the feature names in the exact order fed to the teacher."""
    raise NotImplementedError