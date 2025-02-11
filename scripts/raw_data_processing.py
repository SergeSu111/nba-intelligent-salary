import pandas as pd
import os
import unidecode  # To process special characters
from fuzzywuzzy import process, fuzz

# Set folder path
raw_data_path = "C:/nba-intelligent-salary/data/raw/"

# Get list of all CSV files
all_files = [f for f in os.listdir(raw_data_path) if f.endswith(".csv")]

# Filter player data files and salary data files
player_files = [f for f in all_files if "Salary" not in f]
salary_files = [f for f in all_files if "Salary" in f]

# Read all player data
player_dfs = []
for file in player_files:
    df = pd.read_csv(os.path.join(raw_data_path, file))
    # Remove columns with "Unnamed" in their names and drop the MIN column if it exists
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    if "MIN" in df.columns:
        df.drop(columns=["MIN"], inplace=True)
    df["Season"] = file.replace(".csv", "")  # Add season information
    player_dfs.append(df)

# Read all salary data
salary_dfs = []
for file in salary_files:
    df = pd.read_csv(os.path.join(raw_data_path, file))
    df["Season"] = file.replace("_Salary.csv", "")  # Add season information
    salary_dfs.append(df)

# Merge all seasons' data
player_data = pd.concat(player_dfs, ignore_index=True)
salary_data = pd.concat(salary_dfs, ignore_index=True)

# Rename columns in salary_data for consistency
salary_data.rename(columns={"NAME": "Player", "RK": "Salary_RK"}, inplace=True)

# Remove positional information from salary_data player names (e.g., ", sf")
salary_data["Player"] = salary_data["Player"].str.split(",").str[0].str.strip()

# Standardize player name format
def clean_player_name(name):
    if pd.isna(name):
        return ""
    name = unidecode.unidecode(name).strip().lower()
    # Remove hyphens, dots, single/double quotes, backticks and other special characters
    name = name.replace("-", " ").replace(".", "").replace("'", "").replace('"', "").replace("`", "").replace("´", "")
    return name

player_data["Player"] = player_data["Player"].apply(clean_player_name)
salary_data["Player"] = salary_data["Player"].apply(clean_player_name)

# Remove suffixes like Jr, Sr, II, III, etc.
suffixes = r'\b(jr|sr|ii|iii|iv|v)\b'
player_data["Player"] = player_data["Player"].str.replace(suffixes, '', regex=True).str.strip()
salary_data["Player"] = salary_data["Player"].str.replace(suffixes, '', regex=True).str.strip()

# Process salary data format: remove "$" and commas, convert to float
salary_data.rename(columns={"SALARY": "Salary"}, inplace=True)
salary_data["Salary"] = salary_data["Salary"].astype(str)
salary_data["Salary"] = salary_data["Salary"].str.replace(r"[,\$]", "", regex=True)
salary_data["Salary"] = pd.to_numeric(salary_data["Salary"], errors="coerce")

# Ensure Season columns are of type string
player_data["Season"] = player_data["Season"].astype(str)
salary_data["Season"] = salary_data["Season"].astype(str)

# Standardize spaces in player names (replace multiple spaces with a single space)
player_data["Player"] = player_data["Player"].str.replace(r'\s+', ' ', regex=True).str.strip()
salary_data["Player"] = salary_data["Player"].str.replace(r'\s+', ' ', regex=True).str.strip()

# Merge player data and salary data on Player and Season
merged_data = pd.merge(player_data, salary_data, on=["Player", "Season"], how="left")

# (如果需要模糊匹配，可在这里进行；但若只保留匹配成功的记录，可直接删除缺失薪资的行)

# Remove records with missing Salary (i.e., unmatched players)
final_data = merged_data.dropna(subset=["Salary"]).copy()
print(f"Final dataset records (all players have salary data): {len(final_data)}")

# Save the final merged DataFrame to a CSV file for further analysis
output_path = "C:/nba-intelligent-salary/data/processed/nba_merged_cleaned.csv"
final_data.to_csv(output_path, index=False)
print(f"Final merged data has been saved to {output_path}")

