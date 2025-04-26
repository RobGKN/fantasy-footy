import pandas as pd
import os
from src.utils import compute_value_score, generate_labels, normalize_per_year

# Only includes stats from *_yearly_averages.csv
YEARLY_AVERAGES_FEATURES = [
    'KI', 'MK', 'HB', 'DI', 'GL', 'BH', 'HO', 'TK', 'RB', 'IF', 'CL', 'CG',
    'FF', 'FA', 'BR', 'CP', 'UP', 'CM', 'MI', '1%', 'BO', 'GA'
]

def extract_player_full_features(firstname, lastname, year, output_feather, data_path="sample_data", yearly_averages_csv=None):
    if year < 2019:
        full_name = f"{lastname}, {firstname}"
    else:
        full_name = f"{firstname} {lastname}"
    standardized_name = f"{firstname}_{lastname}"
    
    input_csv = os.path.join(data_path, "afl_fantasy_data", f"afl_fantasy_{year + 1}.csv")
    if yearly_averages_csv is None:
        yearly_averages_csv = os.path.join(data_path, "yearly_averages", f"{standardized_name}_yearly_averages.csv")
    
    df_fantasy = pd.read_csv(input_csv)
    player_fantasy_data = df_fantasy[df_fantasy['Player'] == full_name]
    if player_fantasy_data.empty:
        return None  # Skip if player not found
    
    player_year_fantasy = player_fantasy_data[['Player', 'Prev_Year_Ave', 'Prev_Year_Games', 'Year_Start_Price']].copy()
    player_year_fantasy.rename(columns={
        'Prev_Year_Ave': 'Average_Points',
        'Prev_Year_Games': 'Games_Played',
        'Year_Start_Price': 'Next_Year_Price'
    }, inplace=True)
    player_year_fantasy['Year'] = year
    player_year_fantasy['Next_Year_Price'] = player_year_fantasy['Next_Year_Price'].replace({'\$': '', ',': ''}, regex=True).astype(int)
    player_year_fantasy['Player'] = standardized_name
    
    df_yearly_averages = pd.read_csv(yearly_averages_csv)
    player_yearly_averages = df_yearly_averages[df_yearly_averages['Year'] == year].copy()
    if player_yearly_averages.empty:
        return None  # Skip if stats not found
    
    player_year_stats = player_yearly_averages[['Year'] + YEARLY_AVERAGES_FEATURES].copy()
    player_year_stats['Player'] = standardized_name
    player_year_full_features = pd.merge(
        player_year_fantasy, player_year_stats, on=['Year', 'Player'], how='inner'
    )
    player_year_full_features.reset_index(drop=True).to_feather(output_feather)
    return output_feather

def extract_yearly_player_summaries(year, output_feather, data_path="sample_data"):
    input_csv = os.path.join(data_path, "afl_fantasy_data", f"afl_fantasy_{year + 1}.csv")
    df_fantasy = pd.read_csv(input_csv)
    
    player_summaries = []
    skipped_players = 0
    
    for _, row in df_fantasy.iterrows():
        firstname, lastname = parse_player_name(row['Player'])
        standardized_lastname = ''.join(filter(str.isalpha, lastname))
        yearly_averages_files = [
            f for f in os.listdir(os.path.join(data_path, "yearly_averages"))
            if f.startswith(f"{firstname}_{standardized_lastname}")
        ]
        player_found = False
        for file in yearly_averages_files:
            yearly_averages_csv = os.path.join(data_path, "yearly_averages", file)
            df_yearly_averages = pd.read_csv(yearly_averages_csv)
            if not df_yearly_averages[df_yearly_averages['Year'] == year].empty:
                output_player_feather = f"temp_{firstname}_{lastname}_{year}.feather"
                result = extract_player_full_features(
                    firstname, lastname, year, output_player_feather,
                    data_path=data_path, yearly_averages_csv=yearly_averages_csv
                )
                if result:
                    player_summary = pd.read_feather(output_player_feather)
                    player_summaries.append(player_summary)
                    os.remove(output_player_feather)
                    player_found = True
                break
        if not player_found:
            skipped_players += 1
    
    print(f"[{year}] Skipped {skipped_players} players")
    
    if player_summaries:
        combined_summaries = pd.concat(player_summaries, ignore_index=True)
        combined_summaries.to_feather(output_feather)
    else:
        print(f"[{year}] No player summaries extracted.")

def generate_all_years_feature_files(start_year=2016, end_year=2024, data_path="sample_data", output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    for year in range(start_year, end_year + 1):
        output_file = os.path.join(output_dir, f"afl_fantasy_{year}_summaries.feather")
        print(f"Processing year {year}...")
        extract_yearly_player_summaries(year, output_file, data_path=data_path)
    print("Feature extraction complete for all years.")

def parse_player_name(name):
    """
    Robust player name parsing for AFL Fantasy data:
    Handles both "Lastname, Firstname" and "Firstname Lastname".
    Allows for detection of malformed names (e.g., missing firstname).
    """
    name = name.strip()
    if ',' in name:
        lastname, firstname = [part.strip() for part in name.split(',', 1)]
    else:
        parts = name.split()
        if len(parts) == 1:
            # Log the issue and return a dummy value instead of crashing
            print(f"WARNING: Unexpected single-token player name: '{name}'. Skipping.")
            return None, None
        firstname = parts[0]
        lastname = " ".join(parts[1:])
    return firstname, lastname

def generate_aggregated_3yr_features(start_year=2016, end_year=2024, data_path="sample_data", output_dir="output"):
    import pandas as pd
    import os
    from src.utils import compute_value_score, generate_labels, normalize_per_year

    os.makedirs(output_dir, exist_ok=True)
    all_yearly_files = {}

    # Load all existing yearly summary files into memory
    for year in range(start_year, end_year + 1):
        file_path = os.path.join(output_dir, f"afl_fantasy_{year}_summaries.feather")
        if os.path.exists(file_path):
            all_yearly_files[year] = pd.read_feather(file_path)
        else:
            print(f"WARNING: Missing summary file for year {year}")

    # Identify bad players (missing Average_Points or Games_Played)
    bad_players = set()
    for year_df in all_yearly_files.values():
        bad_rows = year_df[
            year_df['Average_Points'].isnull() | year_df['Games_Played'].isnull()
        ]
        bad_players.update(bad_rows['Player'].unique())

    print(f"Excluding {len(bad_players)} players due to missing critical stats (Games_Played or Average_Points).")
    print("Players being excluded due to missing stats:", sorted(bad_players))

    # Filter them out of all years:
    for year in all_yearly_files:
        all_yearly_files[year] = all_yearly_files[year][~all_yearly_files[year]['Player'].isin(bad_players)]

    aggregated_all_years = []

    for target_year in range(start_year, end_year + 1):
        print(f"Aggregating features for year {target_year}...")
        target_data = all_yearly_files.get(target_year)
        if target_data is None:
            print(f"Skipping {target_year}, no summary file.")
            continue

        aggregated_rows = []
        for _, row in target_data.iterrows():
            player = row['Player']
            year = row['Year']

            last_year = year
            prev_years = [last_year - 1, last_year - 2, last_year - 3]
            pre_dataset_coverage = any(y < start_year for y in prev_years)

            last_year_data = all_yearly_files.get(last_year - 1)
            last_year_row = None
            if last_year_data is not None:
                last_year_row = last_year_data[last_year_data['Player'] == player]

            last_year_games = int(last_year_row['Games_Played'].values[0]) if last_year_row is not None and not last_year_row.empty else 0
            last_year_dnp_flag = int(last_year_row is None or last_year_row.empty or last_year_games == 0)
            played_last_year = 0 if last_year_dnp_flag else 1

            # Per-stat last year averages
            last_year_stats = {}
            if last_year_row is not None and not last_year_row.empty and last_year_games > 0:
                for stat in YEARLY_AVERAGES_FEATURES:
                    last_year_stats[f"last_year_{stat}"] = 0 if pd.isnull(last_year_row[stat].values[0]) else last_year_row[stat].values[0]
            else:
                for stat in YEARLY_AVERAGES_FEATURES:
                    last_year_stats[f"last_year_{stat}"] = 0

            last_year_avg_points = last_year_row['Average_Points'].values[0] if last_year_row is not None and not last_year_row.empty else 0
            next_year_price = row['Next_Year_Price']  # still available in the current year's summary

            # 3-year aggregates
            three_years_data = [
                all_yearly_files.get(y)[all_yearly_files.get(y)['Player'] == player] if all_yearly_files.get(y) is not None else pd.DataFrame()
                for y in prev_years
            ]
            valid_years = [df for df in three_years_data if not df.empty and df['Games_Played'].values[0] > 0]

            if valid_years:
                three_year_avg_points = sum(df['Average_Points'].values[0] for df in valid_years) / len(valid_years)
                three_year_total_games = sum(df['Games_Played'].values[0] for df in valid_years)
                three_year_stat_avgs = {
                    f"3yr_avg_{stat}": sum(df[stat].fillna(0).values[0] for df in valid_years) / len(valid_years)
                    for stat in YEARLY_AVERAGES_FEATURES
                }
            else:
                three_year_avg_points = 0
                three_year_total_games = 0
                three_year_stat_avgs = {f"3yr_avg_{stat}": 0 for stat in YEARLY_AVERAGES_FEATURES}

            aggregated_row = {
                'Player': player,
                'Year': year,
                'last_year_avg_points': last_year_avg_points,
                'last_year_games': last_year_games,
                'last_year_dnp_flag': last_year_dnp_flag,
                '3yr_avg_points': three_year_avg_points,
                '3yr_total_games': three_year_total_games,
                'played_last_year': played_last_year,
                'pre_dataset_coverage_flag': int(pre_dataset_coverage),
                'Next_Year_Price': next_year_price
            }
            aggregated_row.update(last_year_stats)
            aggregated_row.update(three_year_stat_avgs)

            aggregated_rows.append(aggregated_row)

        # Combine into DataFrame for this year
        df_aggregated = pd.DataFrame(aggregated_rows)
        aggregated_all_years.append(df_aggregated)

        output_path = os.path.join(output_dir, f"afl_fantasy_aggregated_3yr_{target_year}.feather")
        df_aggregated.reset_index(drop=True).to_feather(output_path)
        print(f"Saved aggregated features for {target_year} to {output_path}")

    # Combine across all years
    if aggregated_all_years:
        full_aggregated_df = pd.concat(aggregated_all_years, ignore_index=True)

        # NEW: Build future points lookup from AFL Fantasy CSVs (not yearly averages)
        future_points_list = []
        for year in range(start_year, end_year + 1):
            future_csv_path = os.path.join(data_path, "afl_fantasy_data", f"afl_fantasy_{year + 1}.csv")
            if os.path.exists(future_csv_path):
                df_future = pd.read_csv(future_csv_path)
                df_future['Future_Year'] = year + 1
                df_future = df_future.rename(columns={
                    'Player': 'Player',
                    'Prev_Year_Ave': 'future_avg_points'
                })
                future_points_list.append(df_future[['Player', 'Future_Year', 'future_avg_points']])
        future_points_lookup = pd.concat(future_points_list, ignore_index=True)

        # Add "future year" column to aggregated data
        full_aggregated_df['Future_Year'] = full_aggregated_df['Year'] + 1

        #fixup name mismatch with csv before merge
        def standardize_player_name(name):
            firstname, lastname = parse_player_name(name)
            if firstname is None or lastname is None:
                return None  # Handles the malformed case gracefully
            return f"{firstname}_{lastname}"

        future_points_lookup['Player'] = future_points_lookup['Player'].apply(standardize_player_name)
        
        ##sanity check danger
        #example_player = "Patrick_Dangerfield"
        #example_year = 2016
        #future_year = example_year + 1
#
        ## Check if this player exists in future_points_lookup
        #print("sanity check", future_points_lookup[(future_points_lookup['Player'] == example_player) & (future_points_lookup['Future_Year'] == future_year)])
        #print("sanity check", future_points_lookup.head(10))
        #print("check name func Dangerfield, Patrick: ", standardize_player_name("Dangerfield, Patrick"))
        
        # Join future points back onto the current year rows
        full_aggregated_df = pd.merge(
            full_aggregated_df,
            future_points_lookup,
            left_on=['Player', 'Future_Year'],
            right_on=['Player', 'Future_Year'],
            how='left'
        )
        # After your merge:
        missing_join = full_aggregated_df[full_aggregated_df['future_avg_points'].isnull()]

        print(f"Number of rows where the join failed (future_avg_points is NaN): {len(missing_join)}")
        print("Example players where join failed:")
        print(missing_join[['Player', 'Year', 'Future_Year']].head(20))
        
        # Corrected label availability logic
        full_aggregated_df['label_available'] = full_aggregated_df['future_avg_points'].notnull().astype(int)

        # Handle DNP players (listed but zero games → NaN future_avg_points → set to 0)
        full_aggregated_df.loc[
            (full_aggregated_df['label_available'] == 1) & (full_aggregated_df['future_avg_points'].isnull()),
            'future_avg_points'
        ] = 0

        # Only normalize and label rows where label is available
        label_ready_df = full_aggregated_df[full_aggregated_df['label_available'] == 1].copy()

        # Normalize actual future points (the real output) and price for these rows
        label_ready_df = normalize_per_year(label_ready_df, ['future_avg_points', 'Next_Year_Price'])
        label_ready_df = compute_value_score(label_ready_df, points_col='future_avg_points_norm', price_col='Next_Year_Price_norm')
        label_ready_df = generate_labels(label_ready_df, value_score_col='value_score', threshold=0.15)

        # Merge the labeled data back into the full dataset
        full_aggregated_df = full_aggregated_df.merge(
            label_ready_df[['Player', 'Year', 'value_score', 'breakout']],
            on=['Player', 'Year'],
            how='left'
        )

        # Clean up: drop helper column
        full_aggregated_df.drop(columns=['Future_Year'], inplace=True)

        # Overwrite each year's file now with correct labels and flags
        for year in range(start_year, end_year + 1):
            year_df = full_aggregated_df[full_aggregated_df['Year'] == year]
            output_path = os.path.join(output_dir, f"afl_fantasy_aggregated_3yr_{year}.feather")
            year_df.reset_index(drop=True).to_feather(output_path)
            print(f"Re-saved aggregated features with labels (where available) for {year} to {output_path}")

if __name__ == "__main__":
    generate_aggregated_3yr_features(data_path='data', output_dir='output')
    
    
    #generate_all_years_feature_files(data_path = 'data')

    '''
    # Example: extract features for a single player
    firstname = "Dion"
    lastname = "Prestia"
    year = 2016
    output_path = "output/prestia_2016_full_features.feather"
    extract_player_full_features(firstname, lastname, year, output_path, data_path=data_path)
    
    # Example: extract yearly summaries for all players
    year = 2016
    output_path = "output/afl_fantasy_2016_summaries.feather"
    extract_yearly_player_summaries(year, output_path, data_path=data_path) '''
