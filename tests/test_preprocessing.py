import pandas as pd
import os
from src.data_preprocessing import extract_player_full_features
from src.data_preprocessing import extract_yearly_player_summaries
from src.data_preprocessing import parse_player_name

def test_extract_player_full_features(tmp_path):
    # Define input parameters
    firstname = "Dion"
    lastname = "Prestia"
    year = 2016
    
    # Define output Feather file
    output_feather = tmp_path / "prestia_2016_full_features.feather"
    
    # Run the extraction function
    extract_player_full_features(firstname, lastname, year, output_feather)
    
    # Load the output and verify
    result = pd.read_feather(output_feather)
    
    # Hardcoded expected values
    assert len(result) == 1  # Only one entry for Dion Prestia
    assert result["Player"].iloc[0] == "Dion_Prestia"
    assert result["Average_Points"].iloc[0] == 92.93
    assert result["Games_Played"].iloc[0] == 14
    assert result["Next_Year_Price"].iloc[0] == 561000
    assert result["KI"].iloc[0] == 12.29
    assert result["TK"].iloc[0] == 5.36
    assert result["GA"].iloc[0] == 0.36

def test_extract_yearly_player_summaries(tmp_path):
    # Define input parameters
    year = 2016
    
    # Define output Feather file
    output_feather = tmp_path / "afl_fantasy_2016_summaries.feather"
    
    # Run the extraction function
    extract_yearly_player_summaries(year, output_feather)
    
    # Load the output and verify
    result = pd.read_feather(output_feather)
    
    # Verify that Dion Prestia and Gary Ablett are included
    assert len(result) >= 2  # At least two players should be included
    assert "Dion_Prestia" in result["Player"].values
    assert "Gary_Ablett" in result["Player"].values
    
    # Verify some expected values for Dion Prestia
    prestia_row = result[result["Player"] == "Dion_Prestia"].iloc[0]
    assert prestia_row["Average_Points"] == 92.93
    assert prestia_row["Games_Played"] == 14
    assert prestia_row["Next_Year_Price"] == 561000
    
    # Verify some expected values for Gary Ablett
    ablett_row = result[result["Player"] == "Gary_Ablett"].iloc[0]
    assert ablett_row["Games_Played"] > 0  # Ensure he played at least one game

assert parse_player_name("Kennedy, Josh") == ("Josh", "Kennedy")
assert parse_player_name("Josh Kennedy") == ("Josh", "Kennedy")
assert parse_player_name("Josh P. Kennedy") == ("Josh", "P. Kennedy")
assert parse_player_name("Anthony McDonald-Tipungwuti") == ("Anthony", "McDonald-Tipungwuti")
assert parse_player_name("McDonald-Tipungwuti, Anthony") == ("Anthony", "McDonald-Tipungwuti")

