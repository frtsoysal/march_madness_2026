"""
March Madness 2026 — Final Prediction Pipeline
================================================
Evaluation: Log-Loss (NOT Brier Score)
Target: Men's tournament only
Approach: 3-model ensemble with conservative calibration
Based on: Raddar/vilnius-ncaa (2nd place 2025), Mike Kim (4th place 2025)
"""

# %% [markdown]
# # March Madness 2026 — Log-Loss Optimized Prediction Pipeline

# %% Cell 1: Imports & Configuration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import xgboost as xgb
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize, minimize_scalar
import tqdm

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 999)
sns.set_palette("deep")

DATA_DIR = "march-machine-learning-mania-2026"
SEASON_START = 2003  # First season with detailed box scores
CURRENT_SEASON = 2026
CLIP_LOW, CLIP_HIGH = 0.05, 0.95

print("Configuration loaded.")

# %% Cell 2: Load Data
print("Loading data...")
reg_detailed = pd.read_csv(f"{DATA_DIR}/MRegularSeasonDetailedResults.csv")
reg_compact = pd.read_csv(f"{DATA_DIR}/MRegularSeasonCompactResults.csv")
tourney_compact = pd.read_csv(f"{DATA_DIR}/MNCAATourneyCompactResults.csv")
tourney_detailed = pd.read_csv(f"{DATA_DIR}/MNCAATourneyDetailedResults.csv")
seeds = pd.read_csv(f"{DATA_DIR}/MNCAATourneySeeds.csv")
massey = pd.read_csv(f"{DATA_DIR}/MMasseyOrdinals.csv")
teams = pd.read_csv(f"{DATA_DIR}/MTeams.csv")
conferences = pd.read_csv(f"{DATA_DIR}/MTeamConferences.csv")
submission = pd.read_csv(f"{DATA_DIR}/SampleSubmissionStage2.csv")

# Filter to SEASON_START+
reg_detailed = reg_detailed[reg_detailed["Season"] >= SEASON_START]
reg_compact = reg_compact[reg_compact["Season"] >= SEASON_START]
tourney_detailed = tourney_detailed[tourney_detailed["Season"] >= SEASON_START]
seeds = seeds[seeds["Season"] >= SEASON_START]

print(f"Regular season: {len(reg_detailed)} games ({reg_detailed['Season'].min()}-{reg_detailed['Season'].max()})")
print(f"Tournament: {len(tourney_detailed)} games ({tourney_detailed['Season'].min()}-{tourney_detailed['Season'].max()})")
print(f"Seeds: {len(seeds)} entries")
print(f"Massey: {len(massey)} rows")

# %% Cell 3: Data Doubling & Preprocessing
def prepare_data(df):
    """Double dataset: each game from both team perspectives. Normalize for overtime."""
    cols = ["Season", "DayNum", "LTeamID", "LScore", "WTeamID", "WScore", "NumOT",
            "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF",
            "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF"]
    df = df[cols].copy()

    # Overtime adjustment: normalize stats to 40-minute game (NOT TeamIDs)
    adjot = (40 + 5 * df["NumOT"]) / 40
    adj_cols = [c for c in cols if c not in ["Season", "DayNum", "NumOT", "LTeamID", "WTeamID"]]
    for col in adj_cols:
        df[col] = df[col] / adjot

    # Original: W -> T1, L -> T2
    dfswap = df.copy()
    df.columns = [x.replace("W", "T1_").replace("L", "T2_") for x in df.columns]
    dfswap.columns = [x.replace("L", "T1_").replace("W", "T2_") for x in dfswap.columns]

    output = pd.concat([df, dfswap]).reset_index(drop=True)
    output["PointDiff"] = output["T1_Score"] - output["T2_Score"]
    output["win"] = (output["PointDiff"] > 0).astype(int)
    return output

regular_data = prepare_data(reg_detailed)
tourney_data = prepare_data(tourney_detailed)

# Parse seeds
seeds["seed"] = seeds["Seed"].apply(lambda x: int(x[1:3]))
seeds_T1 = seeds[["Season", "TeamID", "seed"]].copy()
seeds_T1.columns = ["Season", "T1_TeamID", "T1_seed"]
seeds_T2 = seeds[["Season", "TeamID", "seed"]].copy()
seeds_T2.columns = ["Season", "T2_TeamID", "T2_seed"]

# Merge seeds onto tournament data
tourney_data = tourney_data[["Season", "T1_TeamID", "T2_TeamID", "PointDiff", "win"]]
tourney_data = pd.merge(tourney_data, seeds_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, seeds_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data["Seed_diff"] = tourney_data["T2_seed"] - tourney_data["T1_seed"]

print(f"Regular data (doubled): {len(regular_data)} rows")
print(f"Tournament data (doubled): {len(tourney_data)} rows")
print(f"Seasons with seeds: {sorted(seeds['Season'].unique())}")

# %% Cell 4: Feature Engineering — Box Score Season Averages
print("Computing box score season averages...")
boxcols = [
    "T1_Score", "T1_FGM", "T1_FGA", "T1_FGM3", "T1_FGA3", "T1_FTM", "T1_FTA",
    "T1_OR", "T1_DR", "T1_Ast", "T1_TO", "T1_Stl", "T1_Blk", "T1_PF",
    "T2_Score", "T2_FGM", "T2_FGA", "T2_FGM3", "T2_FGA3", "T2_FTM", "T2_FTA",
    "T2_OR", "T2_DR", "T2_Ast", "T2_TO", "T2_Stl", "T2_Blk", "T2_PF",
    "PointDiff",
]

ss = regular_data.groupby(["Season", "T1_TeamID"])[boxcols].agg("mean").reset_index()

ss_T1 = ss.copy()
ss_T1.columns = ["T1_avg_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in ss_T1.columns]
ss_T1 = ss_T1.rename({"T1_avg_Season": "Season", "T1_avg_TeamID": "T1_TeamID"}, axis=1)

ss_T2 = ss.copy()
ss_T2.columns = ["T2_avg_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in ss_T2.columns]
ss_T2 = ss_T2.rename({"T2_avg_Season": "Season", "T2_avg_TeamID": "T2_TeamID"}, axis=1)

tourney_data = pd.merge(tourney_data, ss_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, ss_T2, on=["Season", "T2_TeamID"], how="left")

print(f"Box score features added. Tourney data shape: {tourney_data.shape}")

# %% Cell 5: Feature Engineering — Elo Ratings
print("Computing Elo ratings...")

def update_elo(winner_elo, loser_elo, k_factor=100, elo_width=400):
    expected = 1.0 / (1 + 10 ** ((loser_elo - winner_elo) / elo_width))
    change = k_factor * (1 - expected)
    return winner_elo + change, loser_elo - change

base_elo = 1000

# Use compact results for more history; filter to wins only from doubled data
elos_list = []
all_seasons = sorted(set(reg_compact["Season"]))
for season in all_seasons:
    ss_games = reg_compact[reg_compact["Season"] == season]
    team_ids = set(ss_games["WTeamID"]) | set(ss_games["LTeamID"])
    elo = {t: base_elo for t in team_ids}

    for row in ss_games.itertuples(index=False):
        w, l = row.WTeamID, row.LTeamID
        elo[w], elo[l] = update_elo(elo[w], elo[l])

    elo_df = pd.DataFrame({"TeamID": list(elo.keys()), "elo": list(elo.values())})
    elo_df["Season"] = season
    elos_list.append(elo_df)

elos = pd.concat(elos_list)

elos_T1 = elos.rename({"TeamID": "T1_TeamID", "elo": "T1_elo"}, axis=1)
elos_T2 = elos.rename({"TeamID": "T2_TeamID", "elo": "T2_elo"}, axis=1)
tourney_data = pd.merge(tourney_data, elos_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, elos_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data["elo_diff"] = tourney_data["T1_elo"] - tourney_data["T2_elo"]

print(f"Elo ratings computed for {len(all_seasons)} seasons. 2026 teams with Elo: {len(elos[elos['Season']==2026])}")

# %% Cell 6: Feature Engineering — GLM Team Quality
print("Computing GLM team quality scores...")

regular_data["ST1"] = regular_data["Season"].astype(int).astype(str) + "/" + regular_data["T1_TeamID"].astype(int).astype(str)
regular_data["ST2"] = regular_data["Season"].astype(int).astype(str) + "/" + regular_data["T2_TeamID"].astype(int).astype(str)
seeds_T1["ST1"] = seeds_T1["Season"].astype(int).astype(str) + "/" + seeds_T1["T1_TeamID"].astype(int).astype(str)
seeds_T2["ST2"] = seeds_T2["Season"].astype(int).astype(str) + "/" + seeds_T2["T2_TeamID"].astype(int).astype(str)

# Collect tourney teams + teams that beat tourney teams
st = set(seeds_T1["ST1"]) | set(seeds_T2["ST2"])
st = st | set(regular_data.loc[
    (regular_data["T1_Score"] > regular_data["T2_Score"]) & (regular_data["ST2"].isin(st)), "ST1"
])

# For 2026 (no seeds yet): use top 68 by Massey POM rank as proxy tournament teams
massey_2026 = massey[(massey["Season"] == 2026) & (massey["SystemName"] == "POM")]
if len(massey_2026) > 0:
    latest_day = massey_2026["RankingDayNum"].max()
    pom_latest = massey_2026[massey_2026["RankingDayNum"] == latest_day].nsmallest(68, "OrdinalRank")
    for _, row in pom_latest.iterrows():
        st.add(f"2026/{int(row['TeamID'])}")
    # Also add teams that beat these proxy tourney teams in 2026
    st_2026 = {s for s in st if s.startswith("2026/")}
    st = st | set(regular_data.loc[
        (regular_data["Season"] == 2026) &
        (regular_data["T1_Score"] > regular_data["T2_Score"]) &
        (regular_data["ST2"].isin(st_2026)), "ST1"
    ])

def team_quality(season, dt):
    """Compute team quality via OLS on point differential using dummy variables.
    Uses numpy directly for speed instead of statsmodels formula interface."""
    data = dt.loc[dt["Season"] == season].copy()
    if len(data) < 50:
        return pd.DataFrame(columns=["TeamID", "quality", "Season"])

    # Get unique teams (excluding dummy "0000")
    all_teams = sorted(set(data["T1_TeamID"]) | set(data["T2_TeamID"]))
    team_to_idx = {t: i for i, t in enumerate(all_teams)}
    n_teams = len(all_teams)
    n_games = len(data)

    # Build design matrix: each row has +1 for T1 team, -1 for T2 team
    X = np.zeros((n_games, n_teams))
    t1_ids = data["T1_TeamID"].values
    t2_ids = data["T2_TeamID"].values
    y = data["PointDiff"].values

    for i in range(n_games):
        X[i, team_to_idx[t1_ids[i]]] = 1
        X[i, team_to_idx[t2_ids[i]]] = -1

    # OLS: quality = (X'X)^-1 X'y
    try:
        quality_vals = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return pd.DataFrame(columns=["TeamID", "quality", "Season"])

    quality = pd.DataFrame({"TeamID": all_teams, "quality": quality_vals, "Season": season})
    quality = quality[quality["TeamID"] != "0000"]
    quality["TeamID"] = quality["TeamID"].astype(int)
    return quality

dt = regular_data.loc[regular_data["ST1"].isin(st) | regular_data["ST2"].isin(st)].copy()
dt["T1_TeamID"] = dt["T1_TeamID"].astype(str)
dt["T2_TeamID"] = dt["T2_TeamID"].astype(str)
dt.loc[~dt["ST1"].isin(st), "T1_TeamID"] = "0000"
dt.loc[~dt["ST2"].isin(st), "T2_TeamID"] = "0000"

glm_quality = []
seasons_for_glm = sorted(set(seeds["Season"]) | {2026})
for s in tqdm.tqdm(seasons_for_glm, desc="GLM Quality", unit="season"):
    if s >= SEASON_START:
        glm_quality.append(team_quality(s, dt))

glm_quality = pd.concat(glm_quality).reset_index(drop=True)

glm_quality_T1 = glm_quality.rename({"TeamID": "T1_TeamID", "quality": "T1_quality"}, axis=1)
glm_quality_T2 = glm_quality.rename({"TeamID": "T2_TeamID", "quality": "T2_quality"}, axis=1)
tourney_data = pd.merge(tourney_data, glm_quality_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, glm_quality_T2, on=["Season", "T2_TeamID"], how="left")

print(f"GLM quality computed. 2026 teams with quality: {len(glm_quality[glm_quality['Season']==2026])}")

# %% Cell 7: Feature Engineering — Massey Ordinal Composite
print("Computing Massey ordinal composite rankings...")

def compute_massey_features(massey_df, season):
    s = massey_df[massey_df["Season"] == season]
    if len(s) == 0:
        return pd.DataFrame(columns=["TeamID", "massey_median", "massey_mean", "pom_rank", "Season"])

    # Use latest available day for each system
    latest = s.groupby(["SystemName", "TeamID"])["RankingDayNum"].max().reset_index()
    s = s.merge(latest, on=["SystemName", "TeamID", "RankingDayNum"])

    # Composite: median rank across all systems
    composite = s.groupby("TeamID")["OrdinalRank"].agg(["mean", "median"]).reset_index()
    composite.columns = ["TeamID", "massey_mean", "massey_median"]

    # POM rank separately (strongest KenPom proxy)
    pom = s[s["SystemName"] == "POM"][["TeamID", "OrdinalRank"]].rename(columns={"OrdinalRank": "pom_rank"})
    composite = composite.merge(pom, on="TeamID", how="left")
    composite["Season"] = season
    return composite

massey_features = []
for season in tqdm.tqdm(sorted(set(massey["Season"])), desc="Massey", unit="season"):
    massey_features.append(compute_massey_features(massey, season))
massey_features = pd.concat(massey_features)

massey_T1 = massey_features.rename(columns={"TeamID": "T1_TeamID", "massey_median": "T1_massey_median",
                                              "massey_mean": "T1_massey_mean", "pom_rank": "T1_pom_rank"})
massey_T2 = massey_features.rename(columns={"TeamID": "T2_TeamID", "massey_median": "T2_massey_median",
                                              "massey_mean": "T2_massey_mean", "pom_rank": "T2_pom_rank"})
tourney_data = pd.merge(tourney_data, massey_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, massey_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data["massey_diff"] = tourney_data["T1_massey_median"] - tourney_data["T2_massey_median"]
tourney_data["pom_diff"] = tourney_data["T1_pom_rank"] - tourney_data["T2_pom_rank"]

print(f"Massey features added. 2026 teams: {len(massey_features[massey_features['Season']==2026])}")

# %% Cell 8: Feature Engineering — Barttorvik / Adjusted Efficiency Metrics
print("Loading Barttorvik / adjusted efficiency metrics...")

# --- Historical data (2002-2025) from NCAA_Tourney_2002_2025.csv ---
# This file has per-team adjusted metrics (adjoe, adjde, barthag, adjt) for each tournament game
ncaa_hist = pd.read_csv("NCAA_Tourney_2002_2025.csv")

# Extract per-team-season Barttorvik metrics from historical tournament data
# Each row has team1 and team2 metrics; collect unique team-season combos
bt_hist_parts = []
for prefix, id_col in [("team1", "team1_id"), ("team2", "team2_id")]:
    part = ncaa_hist[["season", id_col, f"{prefix}_adjoe", f"{prefix}_adjde",
                       f"{prefix}_tempo", f"{prefix}_adjtempo"]].copy()
    # Compute barthag from adjoe/adjde: barthag = adjoe^11.5 / (adjoe^11.5 + adjde^11.5)
    adjoe = part[f"{prefix}_adjoe"]
    adjde = part[f"{prefix}_adjde"]
    part["barthag"] = adjoe**11.5 / (adjoe**11.5 + adjde**11.5)
    part.columns = ["Season", "TeamID", "bt_adjoe", "bt_adjde", "bt_tempo", "bt_adjtempo", "bt_barthag"]
    bt_hist_parts.append(part)

bt_hist = pd.concat(bt_hist_parts).drop_duplicates(subset=["Season", "TeamID"]).reset_index(drop=True)
bt_hist["TeamID"] = bt_hist["TeamID"].astype(int)
print(f"  Historical Barttorvik: {len(bt_hist)} team-season entries, seasons {bt_hist['Season'].min()}-{bt_hist['Season'].max()}")

# --- 2026 data from barttorvik_2026.csv ---
bt = pd.read_csv("barttorvik_2026.csv")
team_spellings = pd.read_csv(f"{DATA_DIR}/MTeamSpellings.csv", encoding="latin-1")

def normalize_name(name):
    return str(name).lower().strip().replace(".", "").replace("'", "").replace("-", " ").replace("  ", " ")

bt["team_norm"] = bt["team"].apply(normalize_name)
team_spellings["spell_norm"] = team_spellings["TeamNameSpelling"].apply(normalize_name)
teams["name_norm"] = teams["TeamName"].apply(normalize_name)

name_to_id = {}
for _, row in team_spellings.iterrows():
    name_to_id[row["spell_norm"]] = row["TeamID"]
for _, row in teams.iterrows():
    name_to_id[row["name_norm"]] = row["TeamID"]

bt["TeamID"] = bt["team_norm"].map(name_to_id)

unmapped = bt[bt["TeamID"].isna()]["team"].tolist()
if unmapped:
    print(f"  Unmapped 2026 Barttorvik teams ({len(unmapped)}): attempting fuzzy match...")
    kaggle_names = {row["name_norm"]: row["TeamID"] for _, row in teams.iterrows()}
    for bt_name in unmapped:
        norm = normalize_name(bt_name)
        for kn, kid in kaggle_names.items():
            if norm in kn or kn in norm:
                name_to_id[norm] = kid
                break
    bt["TeamID"] = bt["team_norm"].map(name_to_id)
    still_unmapped = bt[bt["TeamID"].isna()]["team"].tolist()
    if still_unmapped:
        print(f"  Still unmapped ({len(still_unmapped)}): {still_unmapped[:10]}...")

bt_2026 = bt[bt["TeamID"].notna()][["TeamID", "adjoe", "adjde", "barthag", "adjt"]].copy()
bt_2026["TeamID"] = bt_2026["TeamID"].astype(int)
bt_2026["Season"] = 2026
bt_2026.columns = ["TeamID", "bt_adjoe", "bt_adjde", "bt_barthag", "bt_adjtempo", "Season"]
print(f"  2026 Barttorvik: {len(bt_2026)} teams mapped")

# --- Combine historical + 2026 ---
bt_all = pd.concat([bt_hist[["Season", "TeamID", "bt_adjoe", "bt_adjde", "bt_barthag", "bt_adjtempo"]],
                     bt_2026[["Season", "TeamID", "bt_adjoe", "bt_adjde", "bt_barthag", "bt_adjtempo"]]],
                    ignore_index=True)
bt_all = bt_all.drop_duplicates(subset=["Season", "TeamID"])

bt_T1 = bt_all.rename(columns={"TeamID": "T1_TeamID", "bt_adjoe": "T1_bt_adjoe", "bt_adjde": "T1_bt_adjde",
                                 "bt_barthag": "T1_bt_barthag", "bt_adjtempo": "T1_bt_adjtempo"})
bt_T2 = bt_all.rename(columns={"TeamID": "T2_TeamID", "bt_adjoe": "T2_bt_adjoe", "bt_adjde": "T2_bt_adjde",
                                 "bt_barthag": "T2_bt_barthag", "bt_adjtempo": "T2_bt_adjtempo"})

# Merge onto tournament training data (historical seasons)
tourney_data = pd.merge(tourney_data, bt_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, bt_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data["bt_adjoe_diff"] = tourney_data["T1_bt_adjoe"] - tourney_data["T2_bt_adjoe"]
tourney_data["bt_adjde_diff"] = tourney_data["T1_bt_adjde"] - tourney_data["T2_bt_adjde"]
tourney_data["bt_barthag_diff"] = tourney_data["T1_bt_barthag"] - tourney_data["T2_bt_barthag"]

# Implied spread from efficiency: (OE_T1 - DE_T2) - (OE_T2 - DE_T1), scaled by average tempo
# This approximates what sportsbooks compute
avg_tempo = (tourney_data["T1_bt_adjtempo"].fillna(68) + tourney_data["T2_bt_adjtempo"].fillna(68)) / 2
tourney_data["bt_implied_spread"] = (
    (tourney_data["T1_bt_adjoe"] - tourney_data["T2_bt_adjde"]) -
    (tourney_data["T2_bt_adjoe"] - tourney_data["T1_bt_adjde"])
) * avg_tempo / 100.0

# Convert implied spread to win probability using normal CDF (σ ≈ 11)
from scipy.stats import norm
tourney_data["bt_implied_prob"] = norm.cdf(tourney_data["bt_implied_spread"] / 11.0)

bt_coverage = tourney_data["bt_adjoe_diff"].notna().sum() / len(tourney_data) * 100
print(f"Barttorvik features merged. Coverage: {bt_coverage:.1f}% of training data")

# %% Cell 9: Feature Engineering — Recent Form & Win Ratios
print("Computing recent form and win ratios...")

def compute_recent_form(regular_data, n_days=14):
    """Last n_days point differential as momentum feature."""
    form_list = []
    for season in sorted(regular_data["Season"].unique()):
        s = regular_data[regular_data["Season"] == season]
        max_day = s["DayNum"].max()
        recent = s[s["DayNum"] >= max_day - n_days]
        form = recent.groupby("T1_TeamID")["PointDiff"].mean().reset_index()
        form.columns = ["TeamID", "recent_form"]
        form["Season"] = season
        form_list.append(form)
    return pd.concat(form_list)

def compute_win_ratios(regular_data):
    """Overall and away win ratios."""
    ratios = []
    for season in sorted(regular_data["Season"].unique()):
        s = regular_data[regular_data["Season"] == season]
        # Overall: from doubled data, win==1 means T1 won
        total = s.groupby("T1_TeamID")["win"].agg(["mean", "count"]).reset_index()
        total.columns = ["TeamID", "win_ratio", "games_played"]
        total["Season"] = season
        ratios.append(total)
    return pd.concat(ratios)

form = compute_recent_form(regular_data)
form_T1 = form.rename(columns={"TeamID": "T1_TeamID", "recent_form": "T1_recent_form"})
form_T2 = form.rename(columns={"TeamID": "T2_TeamID", "recent_form": "T2_recent_form"})

win_ratios = compute_win_ratios(regular_data)
wr_T1 = win_ratios[["Season", "TeamID", "win_ratio"]].rename(columns={"TeamID": "T1_TeamID", "win_ratio": "T1_win_ratio"})
wr_T2 = win_ratios[["Season", "TeamID", "win_ratio"]].rename(columns={"TeamID": "T2_TeamID", "win_ratio": "T2_win_ratio"})

tourney_data = pd.merge(tourney_data, form_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, form_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, wr_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, wr_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data["form_diff"] = tourney_data["T1_recent_form"] - tourney_data["T2_recent_form"]
tourney_data["wr_diff"] = tourney_data["T1_win_ratio"] - tourney_data["T2_win_ratio"]

print(f"Form and win ratio features added. Final tourney shape: {tourney_data.shape}")

# %% Cell 10: Define Feature Sets
# Model 1 (XGB regression): Raddar features + Massey + form + Barttorvik
features_m1 = [
    "T1_seed", "T2_seed", "Seed_diff",
    # Box score averages
    "T1_avg_Score", "T1_avg_FGA", "T1_avg_OR", "T1_avg_DR",
    "T1_avg_Blk", "T1_avg_PF",
    "T1_avg_opponent_FGA", "T1_avg_opponent_Blk", "T1_avg_opponent_PF",
    "T1_avg_PointDiff",
    "T2_avg_Score", "T2_avg_FGA", "T2_avg_OR", "T2_avg_DR",
    "T2_avg_Blk", "T2_avg_PF",
    "T2_avg_opponent_FGA", "T2_avg_opponent_Blk", "T2_avg_opponent_PF",
    "T2_avg_PointDiff",
    # Elo
    "T1_elo", "T2_elo", "elo_diff",
    # GLM Quality
    "T1_quality", "T2_quality",
    # Massey
    "massey_diff", "pom_diff",
    # Barttorvik adjusted efficiency
    "T1_bt_adjoe", "T2_bt_adjoe", "T1_bt_adjde", "T2_bt_adjde",
    "T1_bt_barthag", "T2_bt_barthag",
    "bt_adjoe_diff", "bt_adjde_diff", "bt_barthag_diff",
    # Form
    "form_diff", "wr_diff",
]

# Model 2 (Logistic Regression): diff features only (simpler, stable)
features_m2 = [
    "Seed_diff", "elo_diff", "massey_diff", "pom_diff", "form_diff", "wr_diff",
    "T1_avg_PointDiff", "T2_avg_PointDiff",
    "bt_adjoe_diff", "bt_adjde_diff", "bt_barthag_diff",
]

# Model 3 (XGB classification): Same as M1
features_m3 = features_m1.copy()

# Drop rows with NaN in key features
tourney_clean = tourney_data.dropna(subset=features_m1 + ["PointDiff", "win"]).copy()
print(f"Clean tournament data: {len(tourney_clean)} rows, {tourney_clean['Season'].nunique()} seasons")
print(f"Model 1 features: {len(features_m1)}")
print(f"Model 2 features: {len(features_m2)}")

# %% Cell 11: Model 1 — XGBoost Regression on Point Differential (Raddar)
print("\n=== MODEL 1: XGBoost reg:squarederror on PointDiff ===")

param_m1 = {
    "objective": "reg:squarederror",
    "booster": "gbtree",
    "eta": 0.01,
    "subsample": 0.6,
    "colsample_bynode": 0.8,
    "num_parallel_tree": 2,
    "min_child_weight": 4,
    "max_depth": 4,
    "tree_method": "hist",
    "grow_policy": "lossguide",
    "max_bin": 32,
    "verbosity": 0,
}
num_rounds_m1 = 700

models_m1 = {}
oof_preds_m1, oof_targets_m1, oof_seasons_m1 = [], [], []

for oof_season in sorted(set(tourney_clean.Season)):
    train = tourney_clean[tourney_clean["Season"] != oof_season]
    val = tourney_clean[tourney_clean["Season"] == oof_season]

    dtrain = xgb.DMatrix(train[features_m1].values, label=train["PointDiff"].values)
    dval = xgb.DMatrix(val[features_m1].values)

    models_m1[oof_season] = xgb.train(params=param_m1, dtrain=dtrain, num_boost_round=num_rounds_m1)
    preds = models_m1[oof_season].predict(dval)

    oof_preds_m1.extend(preds)
    oof_targets_m1.extend(val["PointDiff"].values)
    oof_seasons_m1.extend(val["Season"].values)

oof_preds_m1 = np.array(oof_preds_m1)
oof_targets_m1 = np.array(oof_targets_m1)
oof_labels_m1 = (oof_targets_m1 > 0).astype(float)

print(f"M1 OOF predictions: {len(oof_preds_m1)}")

# %% Cell 12: Model 1 — Spline Calibration
print("Calibrating Model 1 with spline...")

t = 25
dat = sorted(zip(oof_preds_m1, oof_labels_m1), key=lambda x: x[0])
pred_sorted, label_sorted = zip(*dat)
spline_model = UnivariateSpline(np.clip(pred_sorted, -t, t), label_sorted, k=5)
oof_probs_m1 = np.clip(spline_model(np.clip(oof_preds_m1, -t, t)), CLIP_LOW, CLIP_HIGH)

ll_m1 = log_loss(oof_labels_m1, oof_probs_m1)
brier_m1 = brier_score_loss(oof_labels_m1, oof_probs_m1)
auc_m1 = roc_auc_score(oof_labels_m1, oof_probs_m1)
print(f"M1 OOF Log-Loss: {ll_m1:.4f}, Brier: {brier_m1:.4f}, AUC: {auc_m1:.4f}")

# Also try isotonic regression
iso_model = IsotonicRegression(out_of_bounds="clip", y_min=CLIP_LOW, y_max=CLIP_HIGH)
iso_model.fit(oof_preds_m1, oof_labels_m1)
oof_probs_m1_iso = iso_model.predict(oof_preds_m1)
ll_m1_iso = log_loss(oof_labels_m1, oof_probs_m1_iso)
print(f"M1 OOF Log-Loss (isotonic): {ll_m1_iso:.4f}")

# Pick the better calibrator
if ll_m1_iso < ll_m1:
    print("  -> Using isotonic calibration (better log-loss)")
    m1_calibrator = "isotonic"
    oof_probs_m1 = oof_probs_m1_iso
    ll_m1 = ll_m1_iso
else:
    print("  -> Using spline calibration (better log-loss)")
    m1_calibrator = "spline"

# %% Cell 13: Model 2 — Logistic Regression on Diff Features
print("\n=== MODEL 2: Logistic Regression ===")

scaler = StandardScaler()
oof_probs_m2 = np.zeros(len(tourney_clean))
oof_idx = 0

models_m2_list = []
for oof_season in sorted(set(tourney_clean.Season)):
    train = tourney_clean[tourney_clean["Season"] != oof_season]
    val = tourney_clean[tourney_clean["Season"] == oof_season]

    X_train = scaler.fit_transform(train[features_m2].values)
    X_val = scaler.transform(val[features_m2].values)

    lr = LogisticRegression(C=1.0, penalty="l2", max_iter=1000, solver="lbfgs")
    lr.fit(X_train, train["win"].values)

    probs = lr.predict_proba(X_val)[:, 1]
    n_val = len(val)
    oof_probs_m2[oof_idx:oof_idx + n_val] = probs
    oof_idx += n_val
    models_m2_list.append((oof_season, lr, scaler.mean_.copy(), scaler.scale_.copy()))

oof_probs_m2 = np.clip(oof_probs_m2, CLIP_LOW, CLIP_HIGH)
ll_m2 = log_loss(oof_labels_m1, oof_probs_m2)
auc_m2 = roc_auc_score(oof_labels_m1, oof_probs_m2)
print(f"M2 OOF Log-Loss: {ll_m2:.4f}, AUC: {auc_m2:.4f}")

# %% Cell 14: Model 3 — XGBoost Classification (Log-Loss Objective)
print("\n=== MODEL 3: XGBoost binary:logistic ===")

param_m3 = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "booster": "gbtree",
    "eta": 0.01,
    "subsample": 0.6,
    "colsample_bynode": 0.8,
    "num_parallel_tree": 2,
    "min_child_weight": 4,
    "max_depth": 3,  # Shallower to avoid overfit
    "tree_method": "hist",
    "grow_policy": "lossguide",
    "max_bin": 32,
    "verbosity": 0,
}
num_rounds_m3 = 500

models_m3 = {}
oof_preds_m3_raw = []

for oof_season in sorted(set(tourney_clean.Season)):
    train = tourney_clean[tourney_clean["Season"] != oof_season]
    val = tourney_clean[tourney_clean["Season"] == oof_season]

    dtrain = xgb.DMatrix(train[features_m3].values, label=train["win"].values)
    dval = xgb.DMatrix(val[features_m3].values)

    models_m3[oof_season] = xgb.train(params=param_m3, dtrain=dtrain, num_boost_round=num_rounds_m3)
    preds = models_m3[oof_season].predict(dval)
    oof_preds_m3_raw.extend(preds)

oof_probs_m3 = np.clip(np.array(oof_preds_m3_raw), CLIP_LOW, CLIP_HIGH)
ll_m3 = log_loss(oof_labels_m1, oof_probs_m3)
auc_m3 = roc_auc_score(oof_labels_m1, oof_probs_m3)
print(f"M3 OOF Log-Loss: {ll_m3:.4f}, AUC: {auc_m3:.4f}")

# %% Cell 15: Ensemble — Optimize Weights
print("\n=== ENSEMBLE OPTIMIZATION ===")

def ensemble_log_loss(weights):
    w = np.array(weights)
    w = w / w.sum()
    blended = w[0] * oof_probs_m1 + w[1] * oof_probs_m2 + w[2] * oof_probs_m3
    blended = np.clip(blended, CLIP_LOW, CLIP_HIGH)
    return log_loss(oof_labels_m1, blended)

result = minimize(ensemble_log_loss, x0=[0.4, 0.3, 0.3],
                  bounds=[(0.05, 0.9)] * 3, method="SLSQP")
optimal_weights = result.x / result.x.sum()

print(f"Optimal weights: M1={optimal_weights[0]:.3f}, M2={optimal_weights[1]:.3f}, M3={optimal_weights[2]:.3f}")

oof_ensemble = (optimal_weights[0] * oof_probs_m1 +
                optimal_weights[1] * oof_probs_m2 +
                optimal_weights[2] * oof_probs_m3)
oof_ensemble = np.clip(oof_ensemble, CLIP_LOW, CLIP_HIGH)

# %% Cell 16: Temperature Scaling
print("Optimizing temperature scaling...")

def neg_log_loss_temp(T):
    logits = np.log(oof_ensemble / (1 - oof_ensemble))
    scaled = 1 / (1 + np.exp(-logits / T))
    scaled = np.clip(scaled, CLIP_LOW, CLIP_HIGH)
    return log_loss(oof_labels_m1, scaled)

temp_result = minimize_scalar(neg_log_loss_temp, bounds=(0.5, 3.0), method="bounded")
optimal_T = temp_result.x

# Apply temperature scaling
logits = np.log(oof_ensemble / (1 - oof_ensemble))
oof_final = 1 / (1 + np.exp(-logits / optimal_T))
oof_final = np.clip(oof_final, CLIP_LOW, CLIP_HIGH)

ll_final = log_loss(oof_labels_m1, oof_final)
print(f"Optimal temperature: {optimal_T:.4f}")
print(f"\n{'='*60}")
print(f"{'Model':<30} {'Log-Loss':>10} {'AUC':>10}")
print(f"{'='*60}")
print(f"{'M1 XGB Regression':.<30} {ll_m1:>10.4f} {auc_m1:>10.4f}")
print(f"{'M2 Logistic Regression':.<30} {ll_m2:>10.4f} {auc_m2:>10.4f}")
print(f"{'M3 XGB Classification':.<30} {ll_m3:>10.4f} {auc_m3:>10.4f}")
print(f"{'Ensemble (weighted)':.<30} {log_loss(oof_labels_m1, oof_ensemble):>10.4f}")
print(f"{'Ensemble + Temp Scaling':.<30} {ll_final:>10.4f}")
print(f"{'Baseline (always 0.5)':.<30} {log_loss(oof_labels_m1, np.full(len(oof_labels_m1), 0.5)):>10.4f}")
print(f"{'='*60}")

# %% Cell 17: Per-Season Log-Loss Diagnostics
print("\nPer-Season OOF Log-Loss:")
seasons_arr = np.array(oof_seasons_m1)
for s in sorted(set(seasons_arr)):
    mask = seasons_arr == s
    s_ll = log_loss(oof_labels_m1[mask], oof_final[mask])
    n_games = mask.sum()
    print(f"  {int(s)}: {s_ll:.4f} ({n_games//2} games)")

# Last-3-folds validation (2023, 2024, 2025) — comparable to leaderboard
last3_mask = np.isin(seasons_arr, [2023, 2024, 2025])
if last3_mask.sum() > 0:
    ll_last3 = log_loss(oof_labels_m1[last3_mask], oof_final[last3_mask])
    auc_last3 = roc_auc_score(oof_labels_m1[last3_mask], oof_final[last3_mask])
    acc_last3 = np.mean((oof_final[last3_mask] > 0.5) == oof_labels_m1[last3_mask])
    brier_last3 = brier_score_loss(oof_labels_m1[last3_mask], oof_final[last3_mask])
    n_last3 = last3_mask.sum() // 2
    print(f"\n  Last 3 Seasons (2023-2025, {n_last3} games):")
    print(f"    Log-Loss: {ll_last3:.4f}")
    print(f"    AUC:      {auc_last3:.4f}")
    print(f"    Accuracy: {acc_last3:.4f}")
    print(f"    Brier:    {brier_last3:.4f}")

# %% Cell 18: Generate 2026 Predictions
print("\n=== GENERATING 2026 PREDICTIONS ===")

# Parse submission file — men's only
X = submission.copy()
X["Season"] = X["ID"].apply(lambda t: int(t.split("_")[0]))
X["T1_TeamID"] = X["ID"].apply(lambda t: int(t.split("_")[1]))
X["T2_TeamID"] = X["ID"].apply(lambda t: int(t.split("_")[2]))
X = X[X["T1_TeamID"] < 2000].copy()  # Men only
X = X[X["Season"] == 2026].copy()
print(f"Men's matchups to predict: {len(X)}")

# Merge all features for 2026
X = pd.merge(X, ss_T1, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, ss_T2, on=["Season", "T2_TeamID"], how="left")
X = pd.merge(X, elos_T1, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, elos_T2, on=["Season", "T2_TeamID"], how="left")
X = pd.merge(X, glm_quality_T1, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, glm_quality_T2, on=["Season", "T2_TeamID"], how="left")
X = pd.merge(X, massey_T1, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, massey_T2, on=["Season", "T2_TeamID"], how="left")
X = pd.merge(X, form_T1, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, form_T2, on=["Season", "T2_TeamID"], how="left")
X = pd.merge(X, wr_T1, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, wr_T2, on=["Season", "T2_TeamID"], how="left")
# Barttorvik 2026 (from bt_T1/bt_T2 which now includes 2026)
X = pd.merge(X, bt_T1, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, bt_T2, on=["Season", "T2_TeamID"], how="left")

# Computed features
X["elo_diff"] = X["T1_elo"] - X["T2_elo"]
X["Seed_diff"] = X.get("T2_seed", 0) - X.get("T1_seed", 0)
X["massey_diff"] = X["T1_massey_median"] - X["T2_massey_median"]
X["pom_diff"] = X["T1_pom_rank"] - X["T2_pom_rank"]
X["form_diff"] = X["T1_recent_form"] - X["T2_recent_form"]
X["wr_diff"] = X["T1_win_ratio"] - X["T2_win_ratio"]
X["bt_adjoe_diff"] = X["T1_bt_adjoe"] - X["T2_bt_adjoe"]
X["bt_adjde_diff"] = X["T1_bt_adjde"] - X["T2_bt_adjde"]
X["bt_barthag_diff"] = X["T1_bt_barthag"] - X["T2_bt_barthag"]
avg_tempo_x = (X["T1_bt_adjtempo"].fillna(68) + X["T2_bt_adjtempo"].fillna(68)) / 2
X["bt_implied_spread"] = (
    (X["T1_bt_adjoe"] - X["T2_bt_adjde"]) -
    (X["T2_bt_adjoe"] - X["T1_bt_adjde"])
) * avg_tempo_x / 100.0
X["bt_implied_prob"] = norm.cdf(X["bt_implied_spread"] / 11.0)

# Seed proxy from POM rank (since actual seeds not yet available)
# Map POM rank to approximate seed: rank 1-4 -> seed 1, 5-8 -> seed 2, etc.
# For teams not in POM, use seed 12 (typical mid-level team)
for prefix in ["T1", "T2"]:
    pom_col = f"{prefix}_pom_rank"
    seed_col = f"{prefix}_seed"
    if seed_col not in X.columns or X[seed_col].isna().all():
        X[seed_col] = ((X[pom_col] - 1) / 4 + 1).clip(1, 16).fillna(12)
    else:
        X[seed_col] = X[seed_col].fillna(((X[pom_col] - 1) / 4 + 1).clip(1, 16)).fillna(12)
X["Seed_diff"] = X["T2_seed"] - X["T1_seed"]

# Fill NaN with median for remaining features
for col in features_m1:
    if X[col].isna().any():
        median_val = tourney_clean[col].median() if col in tourney_clean.columns else 0
        X[col] = X[col].fillna(median_val)

print(f"Features ready. NaN check: {X[features_m1].isna().sum().sum()} NaNs remaining")

# %% Cell 19: Generate Predictions with All LOSO Models
print("Generating predictions...")

# Model 1: XGB regression + calibration
preds_m1_all = []
for oof_season, model in models_m1.items():
    dtest = xgb.DMatrix(X[features_m1].values)
    margin_preds = model.predict(dtest)
    if m1_calibrator == "isotonic":
        probs = iso_model.predict(margin_preds)
    else:
        probs = spline_model(np.clip(margin_preds, -t, t))
    probs = np.clip(probs, CLIP_LOW, CLIP_HIGH)
    preds_m1_all.append(probs)
X["Pred_m1"] = np.mean(preds_m1_all, axis=0)

# Model 2: Logistic Regression (use the last trained scaler/model, or average)
preds_m2_all = []
for oof_season, lr, mean, scale in models_m2_list:
    X_scaled = (X[features_m2].values - mean) / scale
    probs = lr.predict_proba(X_scaled)[:, 1]
    preds_m2_all.append(probs)
X["Pred_m2"] = np.clip(np.mean(preds_m2_all, axis=0), CLIP_LOW, CLIP_HIGH)

# Model 3: XGB classification
preds_m3_all = []
for oof_season, model in models_m3.items():
    dtest = xgb.DMatrix(X[features_m3].values)
    probs = model.predict(dtest)
    preds_m3_all.append(probs)
X["Pred_m3"] = np.clip(np.mean(preds_m3_all, axis=0), CLIP_LOW, CLIP_HIGH)

# Ensemble
X["Pred"] = (optimal_weights[0] * X["Pred_m1"] +
             optimal_weights[1] * X["Pred_m2"] +
             optimal_weights[2] * X["Pred_m3"])

# Temperature scaling
logits = np.log(X["Pred"] / (1 - X["Pred"]))
X["Pred"] = 1 / (1 + np.exp(-logits / optimal_T))
X["Pred"] = np.clip(X["Pred"], CLIP_LOW, CLIP_HIGH)

print(f"Model prediction range: [{X['Pred'].min():.4f}, {X['Pred'].max():.4f}]")
print(f"Model mean prediction: {X['Pred'].mean():.4f}")

# %% Cell 19b: Blend with Championship Futures (goto_conversion approach)
print("\n--- Blending with 2026 championship futures ---")
futures = pd.read_csv("futures_2026.csv")

# Map futures team names to Kaggle TeamIDs
futures["team_norm"] = futures["team"].apply(normalize_name)
futures["TeamID"] = futures["team_norm"].map(name_to_id)

# Manual mapping for common mismatches
manual_futures = {
    "uconn": "Connecticut",
    "miami (fl)": "Miami FL",
    "nc state": "N.C. State",
    "st johns": "St John's",
    "saint marys": "Saint Mary's",
    "saint louis": "Saint Louis",
    "ucf": "UCF",
    "byu": "BYU",
    "smu": "SMU",
    "tcu": "TCU",
    "vcu": "VCU",
    "uc irvine": "UC Irvine",
    "usc": "USC",
    "ucla": "UCLA",
    "lsu": "LSU",
    "liu": "LIU",
    "ole miss": "Ole Miss",
    "miami (oh)": "Miami OH",
}
for fn, tn in manual_futures.items():
    tn_norm = normalize_name(tn)
    if tn_norm in name_to_id:
        name_to_id[fn] = name_to_id[tn_norm]

# Also try partial matching for remaining unmapped
futures["TeamID"] = futures["team_norm"].map(name_to_id)
unmapped_f = futures[futures["TeamID"].isna()]
if len(unmapped_f) > 0:
    kaggle_names = {row["name_norm"]: row["TeamID"] for _, row in teams.iterrows()}
    for idx, row in unmapped_f.iterrows():
        norm = row["team_norm"]
        for kn, kid in kaggle_names.items():
            if norm in kn or kn in norm:
                futures.loc[idx, "TeamID"] = kid
                break

futures_mapped = futures[futures["TeamID"].notna()].copy()
futures_mapped["TeamID"] = futures_mapped["TeamID"].astype(int)
print(f"  Futures mapped: {len(futures_mapped)}/{len(futures)} teams")

# Remove vig using goto_conversion (SE-based: adjusts longshots more, favorites less)
raw_probs = futures_mapped["implied_prob"].values.astype(float)
se = np.sqrt((raw_probs - raw_probs**2.0) / raw_probs)
step = (raw_probs.sum() - 1.0) / se.sum()
fair_probs = raw_probs - (se * step)
# Clip negative probs (extreme longshots) to small floor, then renormalize
fair_probs = np.maximum(fair_probs, 1e-6)
fair_probs = fair_probs / fair_probs.sum()
futures_mapped["norm_prob"] = fair_probs
print(f"  goto_conversion: vig {raw_probs.sum():.4f} -> 1.0000 (step={step:.6f})")

# Create TeamID -> championship probability lookup
champ_prob = dict(zip(futures_mapped["TeamID"], futures_mapped["norm_prob"]))

# Log5 formula: P(A beats B) = p_A * (1 - p_B) / (p_A * (1 - p_B) + p_B * (1 - p_A))
# Where p_A, p_B are championship probabilities
# This gives head-to-head win probability from futures
default_prob = futures_mapped["norm_prob"].min() * 0.5  # very small for unlisted teams

def log5_h2h(t1_id, t2_id):
    p1 = champ_prob.get(t1_id, default_prob)
    p2 = champ_prob.get(t2_id, default_prob)
    if p1 + p2 == 0:
        return 0.5
    return (p1 - p1 * p2) / (p1 + p2 - 2 * p1 * p2)

X["futures_pred"] = X.apply(lambda row: log5_h2h(row["T1_TeamID"], row["T2_TeamID"]), axis=1)
X["futures_pred"] = X["futures_pred"].clip(CLIP_LOW, CLIP_HIGH)

# Blend model prediction with futures (futures are very informative for team strength)
# Weight: model gets more weight since it uses detailed features
FUTURES_BLEND = 0.15  # 15% futures, 85% model (conservative blend)
X["Pred_model"] = X["Pred"].copy()
X["Pred"] = (1 - FUTURES_BLEND) * X["Pred_model"] + FUTURES_BLEND * X["futures_pred"]
X["Pred"] = np.clip(X["Pred"], CLIP_LOW, CLIP_HIGH)

print(f"  Futures blend weight: {FUTURES_BLEND:.0%}")
print(f"  Final prediction range: [{X['Pred'].min():.4f}, {X['Pred'].max():.4f}]")
print(f"  Final mean prediction: {X['Pred'].mean():.4f}")

# Show top team futures probabilities
top_futures = futures_mapped.nlargest(10, "norm_prob")[["team", "avg_odds", "norm_prob"]]
print(f"\n  Top 10 championship probabilities:")
for _, row in top_futures.iterrows():
    print(f"    {row['team']:20s} +{row['avg_odds']:>6d}  {row['norm_prob']:.1%}")

# %% Cell 20: Sanity Checks
print("\n=== SANITY CHECKS ===")

# Seed-matchup pivot (using POM-based proxy seeds)
X["T1_seed_int"] = X["T1_seed"].astype(int)
X["T2_seed_int"] = X["T2_seed"].astype(int)
pivot = X.pivot_table(index="T1_seed_int", columns="T2_seed_int", values="Pred", aggfunc="mean")
print("\nSeed matchup average probabilities (T1 wins):")
# Show key matchups
for s1, s2, expected in [(1, 16, "~0.95"), (1, 2, "~0.55-0.65"), (8, 9, "~0.50")]:
    if s1 in pivot.index and s2 in pivot.columns:
        val = pivot.loc[s1, s2]
        print(f"  Seed {s1} vs Seed {s2}: {val:.4f} (expected {expected})")

# Probability distribution
print(f"\nPrediction distribution:")
print(f"  < 0.10: {(X['Pred'] < 0.10).sum()} matchups")
print(f"  0.10-0.30: {((X['Pred'] >= 0.10) & (X['Pred'] < 0.30)).sum()} matchups")
print(f"  0.30-0.50: {((X['Pred'] >= 0.30) & (X['Pred'] < 0.50)).sum()} matchups")
print(f"  0.50-0.70: {((X['Pred'] >= 0.50) & (X['Pred'] < 0.70)).sum()} matchups")
print(f"  0.70-0.90: {((X['Pred'] >= 0.70) & (X['Pred'] < 0.90)).sum()} matchups")
print(f"  > 0.90: {(X['Pred'] >= 0.90).sum()} matchups")

# %% Cell 21: Write Submission
output = X[["ID", "Pred"]].copy()
output.to_csv("submission_2026.csv", index=False)
print(f"\nSubmission saved: submission_2026.csv ({len(output)} rows)")
print(f"Sample:")
print(output.head(10))

# %% Cell 22: Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. OOF probability distribution
axes[0, 0].hist(oof_final, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
axes[0, 0].set_xlabel("Predicted Probability")
axes[0, 0].set_ylabel("Count")
axes[0, 0].set_title("OOF Ensemble Probability Distribution")
axes[0, 0].axvline(0.5, color="red", linestyle="--", alpha=0.5)

# 2. Calibration plot
from sklearn.calibration import calibration_curve
fraction_pos, mean_predicted = calibration_curve(oof_labels_m1, oof_final, n_bins=10)
axes[0, 1].plot([0, 1], [0, 1], "k--", label="Perfect")
axes[0, 1].plot(mean_predicted, fraction_pos, "s-", color="steelblue", label="Ensemble")
axes[0, 1].set_xlabel("Mean Predicted Probability")
axes[0, 1].set_ylabel("Actual Win Rate")
axes[0, 1].set_title("Calibration Plot")
axes[0, 1].legend()

# 3. Per-season log-loss
season_lls = []
for s in sorted(set(seasons_arr)):
    mask = seasons_arr == s
    s_ll = log_loss(oof_labels_m1[mask], oof_final[mask])
    season_lls.append((int(s), s_ll))
s_df = pd.DataFrame(season_lls, columns=["Season", "Log-Loss"])
axes[1, 0].bar(s_df["Season"].astype(str), s_df["Log-Loss"], color="steelblue")
axes[1, 0].set_xlabel("Season")
axes[1, 0].set_ylabel("Log-Loss")
axes[1, 0].set_title("Per-Season OOF Log-Loss")
axes[1, 0].tick_params(axis="x", rotation=45)
axes[1, 0].axhline(ll_final, color="red", linestyle="--", alpha=0.5, label=f"Avg: {ll_final:.4f}")
axes[1, 0].legend()

# 4. Feature importance (Model 1, first fold)
first_model = list(models_m1.values())[0]
importance = first_model.get_score(importance_type="weight")
imp_df = pd.DataFrame({"Feature": list(importance.keys()), "Weight": list(importance.values())})
imp_df = imp_df.sort_values("Weight", ascending=True).tail(15)
axes[1, 1].barh(imp_df["Feature"], imp_df["Weight"], color="steelblue")
axes[1, 1].set_xlabel("Weight")
axes[1, 1].set_title("Top 15 Feature Importance (XGB M1)")

plt.tight_layout()
plt.savefig("diagnostics_2026.png", dpi=150, bbox_inches="tight")
plt.show()
print("Diagnostics saved to diagnostics_2026.png")

print("\n=== PIPELINE COMPLETE ===")
print(f"Final OOF Log-Loss: {ll_final:.4f}")
print(f"Submission: submission_2026.csv ({len(output)} men's matchups)")
