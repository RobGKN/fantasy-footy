import pandas as pd
import logging
import yaml
import glob
import os
"""
utils.py

This module provides utility functions for data preprocessing and evaluation 
within the AFL Fantasy breakout prediction pipeline. These utilities handle:

1. Normalization of feature columns per season ('Year') to account for inflation, 
   pricing trends, and scoring environment changes across different AFL seasons.
2. Calculation of the value score (normalized fantasy average divided by normalized price),
   which is used to define the breakout label.
3. Breakout label generation based on a quantile threshold (e.g., top 15% of value scores 
   per season), supporting flexible experimentation with different cutoff strategies.
4. Placeholder for additional metric calculations or evaluation tools 
   (e.g., Precision@K, F1 score, ROC-AUC) as the project evolves.

These functions are designed to keep data transformation logic separate from 
the model and training code, ensuring maintainability and reusability across 
different stages of the project.

Usage:
    df = normalize_per_year(df, columns=['Average_Points', 'Next_Year_Price'])
    df = compute_value_score(df)
    df = generate_labels(df, value_score_col='value_score', threshold=0.15)
"""

def normalize_per_year(df, columns, group_col='Year'):
    for col in columns:
        df[f'{col}_norm'] = df.groupby(group_col)[col].transform(lambda x: (x - x.mean()) / x.std())
    return df

def generate_labels(df, value_score_col, threshold=0.15):
    # Example using quantile cutoff
    df['breakout'] = df.groupby('Year')[value_score_col].transform(lambda x: (x >= x.quantile(1 - threshold)).astype(int))
    return df

def compute_value_score(df, points_col='Average_Points_norm', price_col='Next_Year_Price_norm'):
    """
    Computes the value score for breakout labeling as points per normalized price.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        points_col (str): The name of the normalized points column.
        price_col (str): The name of the normalized price column.

    Returns:
        pd.DataFrame: The dataframe with an added 'value_score' column.
    """
    df['value_score'] = df[points_col] / df[price_col]
    return df

def report_and_handle_missing_values(df, critical_cols, stat_cols):
    """
    Reports missing values for each critical and stat column, logs the counts,
    and applies the following rules:
    
    - Drops rows where 'Average_Points' or 'Games_Played' are NaN.
    - Fills other stat-related missing values with 0 (assumes role-dependent absence).
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        critical_cols (list): Columns that must NOT be missing ('Average_Points', 'Games_Played').
        stat_cols (list): Columns where missing values can be filled with 0.

    Returns:
        pd.DataFrame: Cleaned DataFrame with NaNs handled.
    """
    logging.info("Checking for missing values...")
    
    # Report missing counts
    null_summary = df.isnull().sum()
    logging.info("\nMissing value summary:\n" + null_summary.to_string())

    # Drop rows where critical inputs are missing
    before = len(df)
    df = df.dropna(subset=critical_cols)
    after = len(df)
    logging.info(f"Dropped {before - after} rows due to missing 'Average_Points' or 'Games_Played'")

    # Fill stat feature NaNs with 0
    df[stat_cols] = df[stat_cols].fillna(0)
    logging.info(f"Filled missing stat features ({len(stat_cols)} columns) with 0")

    return df

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_all_years(output_dir):
    all_files = glob.glob(f"{output_dir}/*.feather")
    dfs = [pd.read_feather(f) for f in all_files]
    return pd.concat(dfs, ignore_index=True)
