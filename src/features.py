import pandas as pd

def add_team_features(games: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    df = games.copy()

    home_cols = ["GAME_ID", "TEAM_ID_HOME", "PTS_HOME"]
    away_cols = ["GAME_ID", "TEAM_ID_AWAY", "PTS_AWAY"]

    home = df[home_cols].rename(columns={"TEAM_ID_HOME": "TEAM_ID", "PTS_HOME": "PTS"})
    away = df[away_cols].rename(columns={"TEAM_ID_AWAY": "TEAM_ID", "PTS_AWAY": "PTS"})

    combined = pd.concat([home, away])
    combined.sort_values(["TEAM_ID", "GAME_ID"], inplace=True)

    combined["avg_pts_lastN"] = (
        combined.groupby("TEAM_ID")["PTS"]
        .transform(lambda s: s.shift().rolling(window, min_periods=1).mean())
    )
    combined["games_played"] = combined.groupby("TEAM_ID").cumcount()

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


if __name__ == "__main__":
    games = pd.read_csv("../data/processed/2023_24_games.csv")
    team_feats = add_team_features(games)
    full = merge_features(games, team_feats)

    output_path = "../data/processed/2023_24_features.csv"
    full.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")
    print(full.head())

