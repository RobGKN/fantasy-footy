# src/explain/quick_explain.py
"""
Explainability script for the AFL Breakout MLP.

Produces:
  - Teacher permutation importance (JSON + PNG)
  - Partial dependence plots for top features
  - A shallow CART surrogate tree (rules + PNG)
  - Comparison metrics between teacher and surrogate
  - Top-K player lists for both models
  - Markdown summary snippet for documentation

Usage:
  python -m src.explain.quick_explain --config config.yaml \
      --teacher_path output/training/mlp_breakout_model.pt \
      --outdir output/explain/
"""

from __future__ import annotations
import argparse, json, logging, os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch, yaml, matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

from src.distill.teacher_io import (
    load_trained_teacher,
    build_teacher_dataloaders,
    collect_teacher_logits,
    get_feature_names,
)
from src.distill.metrics import spearman_fidelity, topk_overlap, evaluate_team_value_with_existing_pipeline


# ----------------------------
# utilities
# ----------------------------
def _setup_logging():
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def _load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _ensure_outdir(base_outdir: str | None, fallback: str) -> str:
    outdir = base_outdir if base_outdir else fallback
    os.makedirs(outdir, exist_ok=True)
    return outdir

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0; neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[neg]); out[neg] = e / (1.0 + e)
    return out


# ----------------------------
# explainability methods
# ----------------------------
def permutation_importance_teacher(
    teacher: torch.nn.Module,
    X_val: np.ndarray,
    base_logits: np.ndarray,
    feature_names: List[str],
    device: str = "cpu",
    n_repeats: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Shuffle features and measure effect on teacher predictions."""
    rng = np.random.default_rng(seed)
    base = base_logits.reshape(-1)
    drops = []
    with torch.no_grad():
        for j, name in enumerate(feature_names):
            vals = []
            for _ in range(n_repeats):
                Xp = X_val.copy()
                rng.shuffle(Xp[:, j])
                Xp_t = torch.tensor(Xp, dtype=torch.float32, device=device)
                lp = teacher(Xp_t).squeeze(1).cpu().numpy().reshape(-1)
                val = 1.0 - (spearman_fidelity(base, lp) or 0.0)
                vals.append(val)
            drops.append((name, float(np.mean(vals)), float(np.std(vals))))
    df = pd.DataFrame(drops, columns=["feature", "importance", "std"]).sort_values("importance", ascending=False)
    return df.reset_index(drop=True)

def partial_dependence_teacher(
    teacher: torch.nn.Module,
    X_val: np.ndarray,
    feature_names: List[str],
    idx: int,
    device: str = "cpu",
    num_points: int = 11,
) -> pd.DataFrame:
    """Partial dependence for one feature across quantiles."""
    x = X_val[:, idx]
    qs = np.linspace(0.05, 0.95, num_points)
    grid = np.quantile(x, qs)
    preds = []
    with torch.no_grad():
        for v in grid:
            Xg = X_val.copy()
            Xg[:, idx] = v
            Xg_t = torch.tensor(Xg, dtype=torch.float32, device=device)
            logit = teacher(Xg_t).squeeze(1).cpu().numpy()
            preds.append(_sigmoid(logit).mean())
    return pd.DataFrame({"value": grid, "prob": preds})

def save_pdp_plot(df: pd.DataFrame, feature_name: str, out_path: str) -> None:
    plt.figure(figsize=(5, 3))
    plt.plot(df["value"], df["prob"], marker="o")
    plt.xlabel(feature_name); plt.ylabel("P(breakout)")
    plt.title(f"PDP: {feature_name}")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def save_importance_plot(df_imp: pd.DataFrame, out_path: str, top_k: int = 15) -> None:
    top = df_imp.head(top_k)
    plt.figure(figsize=(6, 4))
    plt.barh(top["feature"][::-1], top["importance"][::-1])
    plt.xlabel("Permutation importance (Δ spearman)")
    plt.title("Teacher permutation importance")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def fit_tiny_tree_on_train(X_train: np.ndarray, tlog_train: np.ndarray) -> DecisionTreeRegressor:
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=50, random_state=42)
    tree.fit(X_train, tlog_train.reshape(-1))
    return tree

def export_tree_png(tree: DecisionTreeRegressor, feature_names: List[str], out_path: str) -> None:
    plt.figure(figsize=(10, 6))
    plot_tree(tree, feature_names=feature_names, filled=True, impurity=False,
              proportion=True, rounded=True, fontsize=8)
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()


# ----------------------------
# main
# ----------------------------
def main(args: argparse.Namespace) -> None:
    _setup_logging()
    cfg = _load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    outdir = _ensure_outdir(args.outdir, "output/explain/")

    teacher = load_trained_teacher(args.teacher_path, device=device)
    train_loader, val_loader, val_df = build_teacher_dataloaders(cfg)
    feature_names = get_feature_names()

    train_out = collect_teacher_logits(teacher, train_loader, device=device, return_labels=False)
    val_out = collect_teacher_logits(teacher, val_loader, device=device, return_labels=True)

    X_train, tlog_train = train_out["X"], train_out["logits"]
    X_val, tlog_val, y_val = val_out["X"], val_out["logits"], val_out["y"]
    tprob_val = _sigmoid(tlog_val)

    # permutation importance
    imp_df = permutation_importance_teacher(teacher, X_val, tlog_val, feature_names, device=device)
    imp_df.to_json(os.path.join(outdir, "teacher_importance.json"), orient="records", indent=2)
    save_importance_plot(imp_df, os.path.join(outdir, "teacher_importance.png"))

    # partial dependence (top 4 features)
    top_feats = imp_df["feature"].head(4).tolist()
    for f in top_feats:
        pdp_df = partial_dependence_teacher(teacher, X_val, feature_names, feature_names.index(f), device=device)
        save_pdp_plot(pdp_df, f, os.path.join(outdir, f"pdp_{f}.png"))
        
    # 2D PDP (manual grid search) for last_year_avg_points x 3yr_avg_points
    feat_last = feature_names.index("last_year_avg_points")
    feat_3yr  = feature_names.index("3yr_avg_points")

    # pick quantile grid for each axis
    grid_last = np.linspace(
        np.percentile(X_val[:, feat_last], 5),
        np.percentile(X_val[:, feat_last], 95),
        25
    )
    grid_3yr = np.linspace(
        np.percentile(X_val[:, feat_3yr], 5),
        np.percentile(X_val[:, feat_3yr], 95),
        25
    )

    Z = np.zeros((len(grid_last), len(grid_3yr)))
    with torch.no_grad():
        for i, v_last in enumerate(grid_last):
            for j, v_3yr in enumerate(grid_3yr):
                Xg = X_val.copy()
                Xg[:, feat_last] = v_last
                Xg[:, feat_3yr]  = v_3yr
                Xg_t = torch.tensor(Xg, dtype=torch.float32, device=device)
                logit = teacher(Xg_t).squeeze(1).cpu().numpy()
                Z[i, j] = _sigmoid(logit).mean()

    plt.figure(figsize=(6, 5))
    contour = plt.contourf(grid_last, grid_3yr, Z.T, cmap="viridis", levels=20)
    plt.colorbar(contour, label="P(breakout)")
    plt.xlabel("last_year_avg_points")
    plt.ylabel("3yr_avg_points")
    plt.title("2D Partial Dependence")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pdp_2d_avgpoints.png"), dpi=150)
    plt.close()
    # shallow surrogate tree
    tiny = fit_tiny_tree_on_train(X_train, tlog_train)
    rules_txt = export_text(tiny, feature_names=feature_names, max_depth=3)
    with open(os.path.join(outdir, "tree_rules.txt"), "w") as f: f.write(rules_txt)
    export_tree_png(tiny, feature_names, os.path.join(outdir, "tree.png"))

    # metrics for surrogate vs teacher
    slog_val = tiny.predict(X_val); sprob_val = _sigmoid(slog_val)
    rho = spearman_fidelity(tlog_val, slog_val)
    overlap = topk_overlap(tprob_val, sprob_val, k=int(cfg["distill"]["cart"].get("k_top_picks", 22)))
    tv = evaluate_team_value_with_existing_pipeline(sprob_val, y_val, val_df.copy(),
        k=int(cfg["distill"]["cart"].get("k_top_picks", 22)),
        random_trials=int(cfg["distill"]["cart"].get("random_trials", 100)),
    )

    metrics = {
        "spearman_fidelity_logit": float(rho) if rho is not None else None,
        "topk_overlap": float(overlap),
        "auc_student": float(tv["auc"]),
        "team_value_model": float(tv["model_team_value"]),
        "team_value_random": float(tv["random_team_value"]),
        "tree_params": {"max_depth": 3, "min_samples_leaf": 50, "seed": 42},
    }
    with open(os.path.join(outdir, "metrics.json"), "w") as f: json.dump(metrics, f, indent=2)

    # save top-K lists for teacher vs surrogate
    k_top = int(cfg["distill"]["cart"].get("k_top_picks", 22))
    t_top = val_df.copy(); t_top["teacher_prob"] = tprob_val
    s_top = val_df.copy(); s_top["student_prob"] = sprob_val
    t_top.sort_values("teacher_prob", ascending=False).head(k_top).to_csv(os.path.join(outdir, "teacher_topk.csv"), index=False)
    s_top.sort_values("student_prob", ascending=False).head(k_top).to_csv(os.path.join(outdir, "student_topk.csv"), index=False)

    # readme-style summary
    with open(os.path.join(outdir, "README_explainability.md"), "w") as f:
        f.write(f"""# Explainability Snapshot

Artifacts generated for interpretability of the AFL Breakout model.

- Permutation importance: `teacher_importance.png`
- Partial dependence plots: top 4 features
- Surrogate tree (depth=3): `tree.png` and `tree_rules.txt`
- Teacher vs surrogate examples: `teacher_topk.csv`, `student_topk.csv`

## Metrics (validation)
- Spearman fidelity: {metrics["spearman_fidelity_logit"]:.3f}
- Top-K overlap: {metrics["topk_overlap"]:.3f}
- Team value (surrogate): {metrics["team_value_model"]:.4f}
- Team value (random): {metrics["team_value_random"]:.4f}

Parameters: max_depth=3, min_samples_leaf=50
""")

    logging.info(f"Explainability artifacts written to {outdir}")


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--teacher_path", default="output/training/mlp_breakout_model.pt")
    p.add_argument("--outdir", default=None)
    args = p.parse_args(); main(args)

if __name__ == "__main__":
    cli()
