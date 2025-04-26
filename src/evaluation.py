from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import logging
import torch


def evaluate_roc_auc(preds, labels):
    return roc_auc_score(labels, preds)

def evaluate_team_value(preds, labels, val_df, k=22, random_trials=100):
    val_df = val_df.copy()
    val_df['pred_score'] = preds
    val_df['value'] = val_df['future_avg_points'] / val_df['Next_Year_Price']

    model_team = val_df.sort_values('pred_score', ascending=False).head(k)
    model_team_total_points = model_team['future_avg_points'].sum()
    model_team_total_price = model_team['Next_Year_Price'].sum()
    model_team_value = model_team_total_points / (model_team_total_price / 1000)

    non_rookies = val_df[val_df['last_year_games'] > 0]
    random_team_values = []
    for _ in range(random_trials):
        random_team = non_rookies.sample(n=k, replace=False)
        total_points = random_team['future_avg_points'].sum()
        total_price = random_team['Next_Year_Price'].sum()
        random_team_values.append(total_points / (total_price / 1000))

    random_baseline_value = np.mean(random_team_values)
    return model_team_value, random_baseline_value

def evaluate_model(model, val_loader, val_df):
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for X, y in val_loader:
            preds = model(X).squeeze().numpy()
            val_preds.extend(preds)
            val_labels.extend(y.numpy())

    val_df = val_df.copy()
    val_df['pred_score'] = val_preds
    return val_df, np.array(val_preds), np.array(val_labels)

def report_evaluation_metrics(val_df, preds, labels, k=22, random_trials=100):
    auc = evaluate_roc_auc(preds, labels)
    logging.info(f"Validation ROC-AUC: {auc:.4f}")

    model_team_value, random_team_value = evaluate_team_value(
        preds, labels, val_df, k=k, random_trials=random_trials
    )
    logging.info(f"Model Best 22 Team Value Score (points per $1k): {model_team_value:.4f}")
    logging.info(f"Random Team Baseline Value (points per $1k) (avg of {random_trials} draws): {random_team_value:.4f}")
    
    k_top_picks = 10
    model_team_value, random_team_value = evaluate_team_value(
        preds, labels, val_df, k=k_top_picks, random_trials=random_trials
    )
    logging.info(f"Model Top {k_top_picks} Picks Team Value Score (points per $1k): {model_team_value:.4f}")
    logging.info(f"Player name: {val_df.loc[val_df['pred_score'].nlargest(k_top_picks).index, 'Player'].values}")
    logging.info(f"Random Top {k_top_picks} Picks Team Baseline Value (points per $1k) (avg of {random_trials} draws): {random_team_value:.4f}")
