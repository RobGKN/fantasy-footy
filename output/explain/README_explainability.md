# Explainability Snapshot

Artifacts generated for interpretability of the AFL Breakout model.

- Permutation importance: `teacher_importance.png`
- Partial dependence plots: top 4 features
- Surrogate tree (depth=3): `tree.png` and `tree_rules.txt`
- Teacher vs surrogate examples: `teacher_topk.csv`, `student_topk.csv`

## Metrics (validation)
- Spearman fidelity: 0.808
- Top-K overlap: 0.136
- Team value (surrogate): 0.0917
- Team value (random): 0.0968

Parameters: max_depth=3, min_samples_leaf=50
