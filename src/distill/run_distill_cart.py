# src/distill/run_distill_cart.py
from __future__ import annotations

import argparse, json, logging, os
from typing import Dict
import numpy as np, pandas as pd, torch, yaml

from src.distill.teacher_io import load_trained_teacher, build_teacher_dataloaders, collect_teacher_logits, get_feature_names
from src.distill.cart import CartStudent
from src.distill.metrics import spearman_fidelity, topk_overlap, evaluate_team_value_with_existing_pipeline

def _setup_logging():
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def _load_config(path: str = "config.yaml") -> Dict:
    with open(path, "r") as f: return yaml.safe_load(f)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float); out = np.empty_like(x, dtype=float)
    pos = x >= 0; neg = ~pos; out[pos] = 1.0/(1.0+np.exp(-x[pos])); e = np.exp(x[neg]); out[neg] = e/(1.0+e); return out

def _ensure_outdir(base_outdir: str | None, cfg_outdir: str) -> str:
    outdir = base_outdir if base_outdir else cfg_outdir; os.makedirs(outdir, exist_ok=True); return outdir

def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f: f.write(text)

def main(args: argparse.Namespace) -> None:
    _setup_logging()
    cfg = _load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    distill_cfg = cfg.get("distill", {}).get("cart", {})
    outdir = _ensure_outdir(args.outdir, distill_cfg.get("output_dir", "output/distill/cart/"))

    mode             = distill_cfg.get("mode", "regression")
    max_depth        = int(distill_cfg.get("max_depth", 5))
    min_samples_leaf = int(distill_cfg.get("min_samples_leaf", 50))
    ccp_alpha        = float(distill_cfg.get("ccp_alpha", 0.0))
    seed             = int(distill_cfg.get("seed", 42))
    k_top            = int(distill_cfg.get("k_top_picks", 22))
    random_trials    = int(distill_cfg.get("random_trials", 100))

    wtq   = float(distill_cfg.get("weight_top_quantile", 0.0))
    top_w = float(distill_cfg.get("top_weight", 1.0))
    res_w = float(distill_cfg.get("rest_weight", 1.0))

    top_focus_enabled = bool(distill_cfg.get("top_focus_enabled", True))
    focus_by_k        = bool(distill_cfg.get("focus_by_k", False))
    focus_top_fraction= float(distill_cfg.get("focus_top_fraction", 0.05))
    neg_band_width    = float(distill_cfg.get("neg_band_width", 0.05))
    pos_weight        = float(distill_cfg.get("pos_weight", 10.0))
    neg_weight        = float(distill_cfg.get("neg_weight", 1.0))

    logging.info("=== Distillation: CART surrogate on teacher logits ===")
    logging.info(f"Artifacts will be written to: {outdir}")
    logging.info(f"CART params: mode={mode}, max_depth={max_depth}, min_samples_leaf={min_samples_leaf}, ccp_alpha={ccp_alpha}, seed={seed}")

    teacher = load_trained_teacher(args.teacher_path, device=device)
    train_loader, val_loader, val_df_in_order = build_teacher_dataloaders(cfg)
    feature_names = get_feature_names()

    train_out = collect_teacher_logits(teacher, train_loader, device=device, return_labels=True)
    val_out   = collect_teacher_logits(teacher, val_loader, device=device, return_labels=True)

    X_train, tlog_train = train_out["X"], train_out["logits"]
    X_val,   tlog_val   = val_out["X"],   val_out["logits"]
    y_val               = val_out["y"]

    tprob_train, tprob_val = _sigmoid(tlog_train), _sigmoid(tlog_val)
    logging.info(f"Train size: {X_train.shape[0]}  | Val size: {X_val.shape[0]}")
    logging.info(f"Val   label rate (mean): {float(np.mean(y_val)):.4f}")

    # --- Build K-matched top-focus subset ---
    if top_focus_enabled:
        if focus_by_k:
            pos_count = max(1, int(round(len(X_train) * (k_top / max(1, len(X_val))))))
            thr_pos = np.partition(tprob_train, -pos_count)[-pos_count]
        else:
            thr_pos = np.quantile(tprob_train, 1.0 - focus_top_fraction)
        thr_neg = max(0.0, thr_pos - neg_band_width)

        pos_mask = (tprob_train >= thr_pos)
        neg_mask = (tprob_train >= thr_neg) & (tprob_train < thr_pos)

        X_sub = np.concatenate([X_train[pos_mask], X_train[neg_mask]], axis=0)
        if mode == "classification":
            y_sub = np.concatenate([np.ones(pos_mask.sum(), dtype=int), np.zeros(neg_mask.sum(), dtype=int)], axis=0)
            targets = y_sub
        else:
            z_sub = np.concatenate([tlog_train[pos_mask], tlog_train[neg_mask]], axis=0)
            targets = z_sub

        w_pos = np.full(pos_mask.sum(), pos_weight, dtype=float)
        w_neg = np.full(neg_mask.sum(), neg_weight, dtype=float)
        sample_weight = np.concatenate([w_pos, w_neg], axis=0)

        logging.info(
            f"Top-focus ON (focus_by_k={focus_by_k}): thr_pos={thr_pos:.4f}, thr_neg={thr_neg:.4f}, "
            f"subset pos={pos_mask.sum()} ({pos_mask.mean():.3f}), neg_band={neg_mask.sum()} ({neg_mask.mean():.3f}), "
            f"weights pos={pos_weight}, neg={neg_weight}"
        )
    else:
        X_sub, targets = X_train, (tlog_train if mode != "classification" else (tprob_train > 0.5).astype(int))
        sample_weight = None
        logging.info("Top-focus OFF: using full training set")

    # Optional global weighting (usually OFF when top-focus is on)
    if wtq > 0.0:
        thresh = np.quantile(tprob_train, 1.0 - wtq)
        w_global_full = np.where(tprob_train >= thresh, top_w, res_w).astype(float)
        if top_focus_enabled:
            w_global_sub = np.concatenate([w_global_full[pos_mask], w_global_full[neg_mask]], axis=0)
            sample_weight = sample_weight * w_global_sub if sample_weight is not None else w_global_sub
        else:
            sample_weight = w_global_full
        logging.info(f"Global weighting ON: q={wtq:.2f}, thresh={thresh:.4f}, top_weight={top_w}, rest_weight={res_w}")

    cart_cfg = {"mode": mode, "max_depth": max_depth, "min_samples_leaf": min_samples_leaf, "ccp_alpha": ccp_alpha, "seed": seed}
    student = CartStudent().fit(X_sub, targets, cart_cfg, sample_weight=sample_weight)

    slog_val  = student.predict_logits(X_val)
    sprob_val = _sigmoid(slog_val)

    rho     = spearman_fidelity(tlog_val, slog_val)
    overlap = topk_overlap(tprob_val, sprob_val, k=k_top)
    tv      = evaluate_team_value_with_existing_pipeline(sprob_val, y_val, val_df_in_order.copy(), k=k_top, random_trials=random_trials)

    from src.evaluation import evaluate_roc_auc
    teacher_auc = float(evaluate_roc_auc(tprob_val, y_val))

    t_idx = np.argsort(-tprob_val)[:k_top]
    s_idx = np.argsort(-sprob_val)[:k_top]
    teacher_topk_recall_in_student = len(set(t_idx).intersection(set(s_idx))) / float(k_top)

    rules_txt = student.export_rules(feature_names=feature_names, max_depth=None)
    _write_text(os.path.join(outdir, "rules.txt"), rules_txt)
    student.save(outdir)

    metrics = {
        "spearman_fidelity_logit": float(rho) if rho is not None else None,
        "topk_overlap": float(overlap),
        "teacher_topk_recall_in_student": float(teacher_topk_recall_in_student),
        "auc_student": float(tv["auc"]),
        "auc_teacher": float(teacher_auc),
        "team_value_model": float(tv["model_team_value"]),
        "team_value_random": float(tv["random_team_value"]),
        "cart_params": {"mode": mode, "max_depth": max_depth, "min_samples_leaf": min_samples_leaf, "ccp_alpha": ccp_alpha, "seed": seed},
        "k_top_picks": k_top,
        "random_trials": random_trials,
        "top_focus_enabled": top_focus_enabled,
        "focus_by_k": focus_by_k,
        "focus_top_fraction": focus_top_fraction,
        "neg_band_width": neg_band_width,
        "pos_weight": pos_weight, "neg_weight": neg_weight,
        "weight_top_quantile": wtq, "top_weight": top_w, "rest_weight": res_w,
    }
    with open(os.path.join(outdir, "metrics.json"), "w") as f: json.dump(metrics, f, indent=2)

    topk_df = val_df_in_order.copy(); topk_df["pred_score"] = sprob_val
    cols = [c for c in ["Player","Year","Next_Year_Price","future_avg_points","pred_score"] if c in topk_df.columns]
    topk_df.sort_values("pred_score", ascending=False).head(k_top)[cols].to_csv(os.path.join(outdir, "topk.csv"), index=False)

    logging.info(f"Wrote rules.txt, metrics.json, topk.csv to {outdir}")
    logging.info("=== Distillation complete ===")

def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--teacher_path", default="output/training/mlp_breakout_model.pt")
    p.add_argument("--outdir", default=None)
    args = p.parse_args(); main(args)

if __name__ == "__main__":
    cli()
