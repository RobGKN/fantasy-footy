import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.model import BreakoutPredictorMLP
from src.dataset import AFLFantasyDataset  # Import the dataset directly
from src.evaluation import evaluate_model
from src.utils import load_config, load_all_years

def load_model(config, input_dim):
    model = BreakoutPredictorMLP(
        input_dim=input_dim,
        hidden_dims=config['model']['hidden_dims'],
        input_dropout=config['model']['input_dropout'],
        output_dropout=config['model']['output_dropout']
    )
    model.load_state_dict(torch.load(config['training']['save_path']))
    model.eval()
    return model

def get_top_k_predictions_for_year(df, feature_cols, config, year, k=4):
    val_df = df[df['Year'] == year].copy()
    val_df = val_df.dropna(subset=feature_cols)

    # Directly use your dataset class
    val_dataset = AFLFantasyDataset(val_df, feature_cols)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    model = load_model(config, input_dim=len(feature_cols))
    val_df, preds, labels = evaluate_model(model, val_loader, val_df)

    top_k = val_df.sort_values('pred_score', ascending=False).head(k)
    print(f"Top {k} predicted breakout players for {year}:")
    top_k['future_avg_points'] = top_k['future_avg_points'].round(2)
    top_k['Next_Year_Price'] = top_k['Next_Year_Price'].round(2)
    top_k['pred_score'] = top_k['pred_score'].round(2)
    print(top_k[['Player','Year', 'pred_score', 'future_avg_points', 'Next_Year_Price']])
    return top_k[['Player', 'Year', 'pred_score', 'future_avg_points', 'Next_Year_Price']]

if __name__ == "__main__":
    config = load_config("config.yaml")
    df = load_all_years(config['data']['output_dir'])
    df = df[df['label_available'] == 1]

    feature_cols = [col for col in df.columns if col.startswith('last_year_') or col.startswith('3yr_avg_') or col in ['last_year_games', 'last_year_dnp_flag', 'played_last_year', 'pre_dataset_coverage_flag']]

    test_year = config['data']['test_year']
    top_10 = get_top_k_predictions_for_year(df, feature_cols, config, year=test_year, k=10)