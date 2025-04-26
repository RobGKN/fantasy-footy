import pytest
import pandas as pd

@pytest.fixture
def dummy_dataframe():
    data = {
        'Year': [2020, 2020, 2021, 2021],
        'Average_Points': [90, 110, 95, 105],
        'Next_Year_Price': [700000, 800000, 750000, 850000],
        'Games_Played': [15, 18, 17, 16]
    }
    return pd.DataFrame(data)
