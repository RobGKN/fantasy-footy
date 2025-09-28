# src/cli/check_teacher_and_topk.py
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import yaml

from src.distill.teacher_io import (
    load_trained_teacher,
    build_teacher_dataloaders,
    collect_teacher_logits,
    get_feature_names,
)
from src.evaluation import evaluate_roc_auc, evaluate_team_value
from src.distill.cart import CartStudent  # optional, only if --student_dir is given


def _setup_logging():
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[neg])
    out[neg] = e / (1.0 + e)
    return out


def _topk_overlap(a_scores: np.ndarray, b_scores: np.ndarray, k: int) -> float:
    a_idx = np.argsort(-a_scores)[:k]
    b_idx = np.argsort(-b_scores)[:k]
    return len(set(a_idx).intersection(set(b_idx))) / float(k)


def main(args: argparse.Namespace) -> None:
    _setup_logging()
    cfg = _load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Build split + load teacher
    train_loader, val_loader, val_df = build_teacher_dataloaders(cfg)
    teacher = load_trained_teacher(args.teacher_path, device=device)
    feature_names = get_feature_names()
    logging.info(f"Val rows: {len(val_df)} | First cols: {feature_names[:6]} ...")

    # 2) Teacher predictions
    val_out = collect_teacher_logits(teacher, val_loader, device=device, return_labels=True)
    tlog = val_out["logits"]
    y = val_out["y"]
    tprob = _sigmoid(tlog)

    # 3) Metrics for teacher on *current* split
    auc = float(evaluate_roc_auc(tprob, y))
    model_team_value, rand_team_value = evaluate_team_value(
        tprob, y, val_df.copy(), k=cfg["distill"]["cart"].get("k_top_picks", 22), random_trials=cfg["distill"]["cart"].get("random_trials", 100)
    )
    logging.info(f"[TEACHER] AUC={auc:.4f} | TeamValue={model_team_value:.6f} vs Random={rand_team_value:.6f}")

    # 4) Teacher top-K CSV
    k_top = int(cfg["distill"]["cart"].get("k_top_picks", 22))
    outdir = args.outdir or cfg["distill"]["cart"].get("output_dir", "output/distill/cart/")
    os.makedirs(outdir, exist_ok=True)
    tdf = val_df.copy()
    tdf["teacher_prob"] = tprob
    t_top = tdf.sort_values("teacher_prob", ascending=False).head(k_top)
    cols = [c for c in ["Player", "Year", "Next_Year_Price", "future_avg_points", "teacher_prob"] if c in t_top.columns]
    t_top[cols].to_csv(os.path.join(outdir, "teacher_topk.csv"), index=False)
    logging.info(f"Wrote teacher_topk.csv to {outdir}")

    # 5) Print band thresholds we’d use for top-focus
    #    K-matched positive threshold and a few quantiles for sanity.
    pos_count = max(1, int(round(len(val_out["logits"]) * (k_top / max(1, len(val_out["logits"]))))))
    # same K as val (this is for inspection; training uses train distribution)
    kq = np.partition(tprob, -k_top)[-k_top]
    q90, q95, q97 = np.quantile(tprob, [0.9, 0.95, 0.97])
    logging.info(f"[TEACHER] Val prob quantiles: q90={q90:.4f}, q95={q95:.4f}, q97={q97:.4f}, K-th={kq:.4f}")

    # 6) If a student is supplied, evaluate overlap/fidelity vs *this* teacher
    if args.student_dir:
        student = CartStudent.load(args.student_dir)
        slog = student.predict_logits(val_out["X"])
        sprob = _sigmoid(slog)
        overlap = _topk_overlap(tprob, sprob, k_top)
        logging.info(f"[STUDENT] Top-K overlap vs teacher: {overlap:.4f}")

        # Also save student_topk.csv for quick diff
        sdf = val_df.copy()
        sdf["student_prob"] = sprob
        s_top = sdf.sort_values("student_prob", ascending=False).head(k_top)
        scols = [c for c in ["Player", "Year", "Next_Year_Price", "future_avg_points", "student_prob"] if c in s_top.columns]
        s_top[scols].to_csv(os.path.join(outdir, "student_topk.csv"), index=False)
        logging.info(f"Wrote student_topk.csv to {outdir}")

        # Join on Player to see mismatches (best-effort; names assumed unique in val year)
        if "Player" in t_top and "Player" in s_top:
            tset = set(t_top["Player"].tolist()); sset = set(s_top["Player"].tolist())
            only_teacher = sorted(tset - sset)
            only_student = sorted(sset - tset)
            dump = {
                "only_teacher": only_teacher,
                "only_student": only_student,
            }
            with open(os.path.join(outdir, "topk_diff.json"), "w") as f:
                json.dump(dump, f, indent=2)
            logging.info(f"Wrote topk_diff.json ({len(only_teacher)} only-teacher, {len(only_student)} only-student)")
    else:
        logging.info("No --student_dir provided; skipped student overlap checks.")


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--teacher_path", default="output/training/mlp_breakout_model.pt")
    p.add_argument("--student_dir", default=None, help="Directory containing tree.pkl/meta.json to compare against")
    p.add_argument("--outdir", default=None)
    args = p.parse_args()
    main(args)


if __name__ == "__main__":
    cli()
