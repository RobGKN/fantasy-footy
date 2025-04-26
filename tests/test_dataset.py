import torch
import pandas as pd
from src.dataset import AFLFantasyDataset

def test_dataset_shape():
    df = pd.DataFrame({
        'Average_Points': [90, 100],
        'Games_Played': [15, 17],
        'Next_Year_Price': [700000, 800000],
        'breakout': [1, 0]
    })
    feature_cols = ['Average_Points', 'Games_Played', 'Next_Year_Price']
    dataset = AFLFantasyDataset(df, feature_cols)
    x, y = dataset[0]
    assert x.shape[0] == len(feature_cols)
    assert isinstance(y.item(), float)
