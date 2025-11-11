import pandas as pd

def add_team_features(games: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    df = games.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    home_cols = ["GAME_ID", "TEAM_ID_HOME", "PTS_HOME", "GAME_DATE"]
    away_cols = ["GAME_ID", "TEAM_ID_AWAY", "PTS_AWAY", "GAME_DATE"]

    home = df[home_cols].rename(columns={"TEAM_ID_HOME": "TEAM_ID", "PTS_HOME": "PTS"})
    away = df[away_cols].rename(columns={"TEAM_ID_AWAY": "TEAM_ID", "PTS_AWAY": "PTS"})

    combined = pd.concat([home, away])
    combined.sort_values(["TEAM_ID", "GAME_ID"], inplace=True)

    combined["avg_pts_lastN"] = (
        combined.groupby("TEAM_ID")["PTS"]
        .transform(lambda s: s.shift().rolling(window, min_periods=1).mean())
    )
    combined["games_played"] = combined.groupby("TEAM_ID").cumcount()
    combined["rest_days"] = combined.groupby("TEAM_ID")["GAME_DATE"].diff().dt.days.fillna(0)
    combined["avg_pts_for"] = combined.groupby("TEAM_ID")["PTS"].transform(
        lambda s: s.shift().rolling(window, min_periods=1).mean())
    combined["avg_pts_against"] = (
        combined.groupby("TEAM_ID")["PTS"]
        .transform(lambda s: s.shift().rolling(window, min_periods=1).mean())
    )
    return combined

def merge_features(main_df, features_df):
    merged = main_df.merge(
        features_df.add_suffix("_HOME"),
        left_on=["GAME_ID", "TEAM_ID_HOME"],
        right_on=["GAME_ID_HOME", "TEAM_ID_HOME"],
        how="left"
    ).merge(
        features_df.add_suffix("_AWAY"),
        left_on=["GAME_ID", "TEAM_ID_AWAY"],
        right_on=["GAME_ID_AWAY", "TEAM_ID_AWAY"],
        how="left"
    )
    return merged

def add_elo_ratings(games, k = 20, start_rating = 1500):
    elo = {}
    home_elos, away_elos = [], []

    for _, row in games.sort_values("GAME_DATE").iterrows():
        h, a = row["TEAM_ID_HOME"], row["TEAM_ID_AWAY"]
        elo.setdefault(h, start_rating)
        elo.setdefault(a, start_rating)

        expected_home = 1 / (1 + 10 ** ((elo[a] - elo[h]) / 400))
        outcome = 1 if row["home_win"] == 1 else 0

        home_elos.append(elo[h])
        away_elos.append(elo[a])

        elo[h] += k * (outcome - expected_home)
        elo[a] += k * ((1 - outcome) - (1 - expected_home))

    games["elo_HOME"] = home_elos
    games["elo_AWAY"] = away_elos
    return games


if __name__ == "__main__":
    games = pd.read_csv("../data/processed/2023_24_games.csv")
    team_feats = add_team_features(games)
    full = merge_features(games, team_feats)
    full["rest_diff"] = full["rest_days_HOME"] - full["rest_days_AWAY"]
    # full["off_def_diff"] = full["avg_pts_for_HOME"] - full["avg_pts_against_AWAY"]
    # full["win_pct_diff"] = full["win_pct_lastN_HOME"] - full["win_pct_lastN_AWAY"]
    # full["net_rating_diff"] = (
    #     (full["avg_pts_for_HOME"] - full["avg_pts_against_AWAY"])
    #     - (full["avg_pts_for_AWAY"] - full["avg_pts_for_HOME"])
    # )
    full = add_elo_ratings(full)
    output_path = "../data/processed/2023_24_features.csv"
    full.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")
    print(full.head())

