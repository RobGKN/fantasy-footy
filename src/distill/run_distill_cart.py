"""
CLI flow:
1) Load config and teacher (teacher_io.load_trained_teacher)
2) Build train/val loaders and get aligned val_df
3) Collect teacher logits on train and val (teacher_io.collect_teacher_logits)
4) Fit CartStudent on train (X, logits)
5) Predict student logits on val; convert to probs via sigmoid
6) Compute:
   - AUC (existing code)
   - Spearman fidelity (metrics.spearman_fidelity) vs teacher logits on val
   - Top-k overlap vs teacher probs on val
   - Team-value metrics via your existing pipeline
7) Export:
   - rules.txt (cart.export_rules)
   - tree.pkl + meta.json (cart.save)
   - metrics.json
   - topk.csv (val_df with pred_score probs, sorted head(k))
"""
import argparse
from typing import Dict

def main(args: argparse.Namespace) -> None:
    raise NotImplementedError

def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--teacher_path", default="output/training/mlp_breakout_model.pt")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    cli()