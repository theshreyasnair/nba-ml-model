import time, pandas as pd
from nba_api.stats.endpoints import leaguegamelog

def fetch_season_logs(season: str, season_type = "Regular Season", save = True) -> pd.DataFrame:
    r = leaguegamelog.LeagueGameLog(season = season, season_type_all_star= season_type)
    time.sleep(0.6)
    df = r.get_data_frames()[0]
    if save:
        df.to_csv(f"../data/raw/{season.replace('-','_')}.csv", index=False)

    return df

if __name__ == "__main__":
    df = fetch_season_logs("2023-24")
    print(df.head())

