from src.dataset import create_dataloaders
from src.model import BreakoutPredictorMLP
from src.train import train_model
import pandas as pd
import torch
from src.evaluation import evaluate_model, report_evaluation_metrics
import yaml
import logging
import os

# ------------------------------
# Config and Logging Setup
# ------------------------------
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

config = load_config("config.yaml")
setup_logging(config['logging']['log_file'])
logging.info("Starting AFL Fantasy Breakout Model Training Pipeline")
logging.info("Loaded configuration:")
for section, params in config.items():
    logging.info(f"{section}: {params}")

# ------------------------------
# Load Aggregated Data
# ------------------------------
def load_all_years(output_dir):
    import glob
    all_files = glob.glob(f"{output_dir}/*.feather")
    dfs = [pd.read_feather(f) for f in all_files]
    return pd.concat(dfs, ignore_index=True)

logging.info("Loading data...")
df = load_all_years(config['data']['output_dir'])
logging.info(f"Loaded {len(df)} player-season records")

# ------------------------------
# Filter Labeled Data Only
# ------------------------------
logging.info("Filtering to labeled data (label_available == 1)...")
df = df[df['label_available'] == 1]
logging.info(f"Filtered to {len(df)} labeled player-season records")


# ------------------------------
# (OLD CODE - NOT USED FOR AGGREGATED DATA, PRESERVED FOR REFERENCE)
# Normalizing and labeling logic (handled during aggregation now)
# logging.info("Normalizing features and generating labels...")
# df = normalize_per_year(df, config['data']['normalize_columns'])
# df = compute_value_score(df)
# df = generate_labels(df, 'value_score', threshold=config['data']['label_threshold'])
# critical_cols = ['Average_Points', 'Games_Played']
# stat_cols = [
#     'KI', 'MK', 'HB', 'DI', 'GL', 'BH', 'HO', 'TK', 'RB', 'IF', 'CL', 'CG',
#     'FF', 'FA', 'BR', 'CP', 'UP', 'CM', 'MI', '1%', 'BO', 'GA'
# ]
# df = report_and_handle_missing_values(df, critical_cols, stat_cols)

# ------------------------------
# Prepare Datasets (NEW AGGREGATED FEATURE SET)
# ------------------------------
feature_cols = [col for col in df.columns 
                if col.startswith('last_year_') 
                or col.startswith('3yr_avg_') 
                or col in ['last_year_games', 'last_year_dnp_flag', 
                           'played_last_year', 'pre_dataset_coverage_flag']]

train_df = df[df['Year'] < config['data']['test_year']]
val_df   = df[df['Year'] == config['data']['test_year']]
# NaN CLEANING - AFTER selecting features
# ------------------------------
logging.info("Checking for remaining NaNs in selected features...")
nan_counts_train = train_df[feature_cols].isnull().sum()
nan_features_train = nan_counts_train[nan_counts_train > 0]
if not nan_features_train.empty:
    logging.warning(f"NaNs in training features:\n{nan_features_train}")
    before = len(train_df)
    train_df = train_df.dropna(subset=feature_cols)
    after = len(train_df)
    logging.info(f"Dropped {before - after} rows from training due to NaNs in selected features.")

nan_counts_val = val_df[feature_cols].isnull().sum()
nan_features_val = nan_counts_val[nan_counts_val > 0]
if not nan_features_val.empty:
    logging.warning(f"NaNs in validation features:\n{nan_features_val}")
    before = len(val_df)
    val_df = val_df.dropna(subset=feature_cols)
    after = len(val_df)
    logging.info(f"Dropped {before - after} rows from validation due to NaNs in selected features.")

# ------------------------------
train_loader, val_loader = create_dataloaders(train_df, val_df, feature_cols, batch_size=config['training']['batch_size'])

logging.info(f"Training set size: {len(train_df)}")
logging.info(f"Validation (test_year={config['data']['test_year']}) set size: {len(val_df)}")



# ------------------------------
# Train Model
# ------------------------------
input_dim = len(feature_cols)
logging.info(f"Starting model training with input dimension {input_dim}...")
model = train_model(train_loader, val_loader, input_dim=input_dim, config=config)
torch.save(model.state_dict(), config['training']['save_path'])
logging.info(f"Model saved to {config['training']['save_path']}")

# ------------------------------
# Evaluation
# ------------------------------
logging.info("Evaluating on holdout (validation) set...")
val_df, preds, labels = evaluate_model(model, val_loader, val_df)
report_evaluation_metrics(val_df, preds, labels, k=22, random_trials=100)