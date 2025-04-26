import pandas as pd
import glob
import os

def query_player_across_all_files(player_name, data_dir="output"):
    # Find all summary and aggregated feather files
    summary_files = glob.glob(os.path.join(data_dir, "afl_fantasy_*_summaries.feather"))
    aggregated_files = glob.glob(os.path.join(data_dir, "afl_fantasy_aggregated_3yr_*.feather"))
    
    print(f"Checking {len(summary_files)} summary files and {len(aggregated_files)} aggregated files...")

    print("\n===== SUMMARY FILES (raw yearly data) =====")
    for file_path in sorted(summary_files):
        df = pd.read_feather(file_path)
        player_rows = df[df['Player'] == player_name]
        if not player_rows.empty:
            print(f"\nFound in {os.path.basename(file_path)}:")
            print(player_rows)

    print("\n===== AGGREGATED 3-YR FILES =====")
    for file_path in sorted(aggregated_files):
        df = pd.read_feather(file_path)
        player_rows = df[df['Player'] == player_name]
        if not player_rows.empty:
            print(f"\nFound in {os.path.basename(file_path)}:")
            print(player_rows)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query player data across all summary and aggregated files.")
    parser.add_argument("player_name", type=str, help="Player name in standardized format (e.g., Tom_Doedee)")
    parser.add_argument("--data_dir", type=str, default="output", help="Directory containing feather files (default: output)")

    args = parser.parse_args()

    query_player_across_all_files(args.player_name, data_dir=args.data_dir)