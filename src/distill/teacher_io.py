# src/distill/teacher_io.py
"""
Utilities for interfacing with the trained MLP teacher:
- load_trained_teacher: load BreakoutPredictorMLP weights and return model.eval()
- build_teacher_dataloaders: construct train/val loaders consistent with your training split
- collect_teacher_logits: run the teacher to collect X/logits/(optional) y
- get_feature_names: return the exact feature order used by the teacher
"""

from typing import Dict, Tuple, List, Optional
import os
import glob
import logging
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import your project modules
from src.dataset import create_dataloaders
from src.model import BreakoutPredictorMLP


# ------------------------------
# Internal helpers mirrored from run_training.py
# ------------------------------
def _load_config(config_path: str = "config.yaml") -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _load_all_years(output_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(output_dir, "*.feather")))
    if not files:
        raise FileNotFoundError(f"No .feather files found under {output_dir}")
    dfs = [pd.read_feather(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def _compute_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Replicates feature selection logic used in run_training.py.
    Preserves the column order from the DataFrame.
    """
    base_includes = {
        "last_year_games",
        "last_year_dnp_flag",
        "played_last_year",
        "pre_dataset_coverage_flag",
    }
    feature_cols = [
        col
        for col in df.columns
        if col.startswith("last_year_")
        or col.startswith("3yr_avg_")
        or col in base_includes
    ]
    return feature_cols


def _dropna_in_selected_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    nan_counts = df[feature_cols].isnull().sum()
    nan_features = nan_counts[nan_counts > 0]
    if not nan_features.empty:
        logging.warning(f"NaNs in selected features:\n{nan_features}")
        before = len(df)
        df = df.dropna(subset=feature_cols)
        after = len(df)
        logging.info(f"Dropped {before - after} rows due to NaNs in selected features.")
    return df


# ------------------------------
# Public API
# ------------------------------
def load_trained_teacher(path: str, device: str = "cpu") -> torch.nn.Module:
    """
    Load BreakoutPredictorMLP with weights at `path` and return in eval mode.
    Uses config.yaml and aggregated features to reconstruct input_dim & architecture.
    """
    cfg = _load_config("config.yaml")
    # Build a quick DF to derive input_dim consistent with training
    df_all = _load_all_years(cfg["data"]["output_dir"])
    df_all = df_all[df_all["label_available"] == 1]
    feature_cols = _compute_feature_cols(df_all)
    input_dim = len(feature_cols)

    hidden_dims = cfg["model"]["hidden_dims"]
    model = BreakoutPredictorMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        input_dropout=cfg["model"].get("input_dropout", 0.0),
        output_dropout=cfg["model"].get("output_dropout", 0.0),
    )

    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def build_teacher_dataloaders(
    config: Dict,
) -> Tuple[DataLoader, DataLoader, pd.DataFrame]:
    """
    Build train/val dataloaders aligned with the teacher’s training split.
    Mirrors run_training.py logic to avoid any data leakage or ordering issues.
    Returns: (train_loader, val_loader, val_df_in_order)
    """
    logging.info("Loading aggregated data for distillation...")
    df = _load_all_years(config["data"]["output_dir"])
    logging.info(f"Loaded {len(df)} player-season records (all years)")

    logging.info("Filtering to labeled data (label_available == 1)...")
    df = df[df["label_available"] == 1]
    logging.info(f"Filtered to {len(df)} labeled player-season records")

    # Feature selection (same as run_training)
    feature_cols = _compute_feature_cols(df)

    # Train/val split by Year
    train_df = df[df["Year"] < config["data"]["test_year"]].copy()
    val_df = df[df["Year"] == config["data"]["test_year"]].copy()

    # Clean NaNs in selected features (mirrors run_training)
    logging.info("Checking for remaining NaNs in selected features (train/val)...")
    train_df = _dropna_in_selected_features(train_df, feature_cols)
    val_df = _dropna_in_selected_features(val_df, feature_cols)

    # Create loaders with the same utility your training used
    batch_size = config["training"]["batch_size"]
    train_loader, val_loader = create_dataloaders(
        train_df, val_df, feature_cols, batch_size=batch_size
    )

    # Ensure val_df aligns row-wise to the iteration order of val_loader
    # (create_dataloaders should preserve order for validation; keep a copy here)
    val_df_in_order = val_df.reset_index(drop=True)

    return train_loader, val_loader, val_df_in_order


def collect_teacher_logits(
    teacher: torch.nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    return_labels: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Run `teacher` over `loader` and collect features and outputs.
    Supports batches shaped as (X, y) or (X, y, idx).
    """
    teacher.eval()
    xs, zs, ys, idxs = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            # Unpack (X, y) or (X, y, idx)
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    X, y = batch
                    idx = None
                elif len(batch) >= 3:
                    X, y, idx = batch[0], batch[1], batch[2]
                else:
                    raise ValueError("Unexpected batch structure from DataLoader.")
            else:
                raise ValueError("Expected batch to be a (X, y) tuple.")

            X = X.to(device)
            logits = teacher(X).squeeze(1)  # [B]
            xs.append(X.detach().cpu().numpy())
            zs.append(logits.detach().cpu().numpy())

            if return_labels and y is not None:
                ys.append(y.detach().cpu().numpy())
            if return_labels and 'idx' in locals() and idx is not None:
                idxs.append(idx.detach().cpu().numpy())

    out: Dict[str, np.ndarray] = {
        "X": np.concatenate(xs, axis=0) if xs else np.empty((0,)),
        "logits": np.concatenate(zs, axis=0) if zs else np.empty((0,)),
    }
    if return_labels and ys:
        out["y"] = np.concatenate(ys, axis=0)
    if return_labels and idxs:
        out["index"] = np.concatenate(idxs, axis=0)

    return out


def get_feature_names() -> List[str]:
    """
    Return the teacher's feature names in the exact order used at training time.
    Derived from the aggregated dataset using the same selection rule as run_training.py.
    """
    cfg = _load_config("config.yaml")
    df_all = _load_all_years(cfg["data"]["output_dir"])
    df_all = df_all[df_all["label_available"] == 1]
    feature_cols = _compute_feature_cols(df_all)
    return feature_cols
