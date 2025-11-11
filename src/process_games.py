import pandas as pd

def process_games(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["HOME_GAME"] = df["MATCHUP"].str.contains("vs")
    df["HOME_TEAM"] = df["MATCHUP"].apply(lambda x: x.split(" vs. ")[0] if "vs." in x else x.split(" @ ")[1])
    df["AWAY_TEAM"] = df["MATCHUP"].apply(lambda x: x.split(" vs. ")[1] if "vs." in x else x.split(" @ ")[0])
    df["WIN"] = df["WL"].eq("W").astype(int)

    home = df[df["HOME_GAME"]][["GAME_ID", "GAME_DATE", "TEAM_ID", "PTS", "WIN"]].rename(
        columns={"TEAM_ID": "TEAM_ID_HOME", "PTS": "PTS_HOME", "WIN": "WIN_HOME"}
    )
    away = df[~df["HOME_GAME"]][["GAME_ID", "GAME_DATE", "TEAM_ID", "PTS", "WIN"]].rename(
        columns={"TEAM_ID": "TEAM_ID_AWAY", "PTS": "PTS_AWAY", "WIN": "WIN_AWAY"}
    )

    merged = home.merge(away, on=["GAME_ID", "GAME_DATE"])
    merged["home_win"] = (merged["PTS_HOME"] > merged["PTS_AWAY"]).astype(int)
    return merged

if __name__ == "__main__":
    df = pd.read_csv("../data/raw/2023_24.csv")
    games = process_games(df)
    games.to_csv("../data/processed/2023_24_games.csv", index=False)
    print(games.head())