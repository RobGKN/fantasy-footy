from src.utils import normalize_per_year, compute_value_score, generate_labels

def test_normalization(dummy_dataframe):
    df = normalize_per_year(dummy_dataframe, ['Average_Points', 'Next_Year_Price'])
    assert 'Average_Points_norm' in df.columns
    assert 'Next_Year_Price_norm' in df.columns
    assert not df['Average_Points_norm'].isnull().any()

def test_value_score(dummy_dataframe):
    df = normalize_per_year(dummy_dataframe, ['Average_Points', 'Next_Year_Price'])
    df = compute_value_score(df)
    assert 'value_score' in df.columns
    assert not df['value_score'].isnull().any()

def test_generate_labels(dummy_dataframe):
    df = normalize_per_year(dummy_dataframe, ['Average_Points', 'Next_Year_Price'])
    df = compute_value_score(df)
    df = generate_labels(df, 'value_score', threshold=0.5)  # 50% quantile for testing
    assert 'breakout' in df.columns
    assert set(df['breakout'].unique()).issubset({0, 1})
