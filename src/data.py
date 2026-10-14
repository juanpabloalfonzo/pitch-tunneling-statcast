from pathlib import Path
import pandas as pd
import pybaseball


REQUIRED_COLUMNS = [
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "pitch_type",
]


def _lookup_player_id(pitcher_name: str) -> int:
    name_parts = pitcher_name.strip().split()
    if len(name_parts) < 2:
        raise ValueError(
            f"Pitcher name must include at least first and last name: '{pitcher_name}'"
        )

    first_name = name_parts[0]
    last_name = " ".join(name_parts[1:])

    lookup_result = pybaseball.playerid_lookup(last_name, first_name)
    if lookup_result.empty:
        raise ValueError(f"Player '{pitcher_name}' not found.")

    key_mlbam = lookup_result.iloc[0]["key_mlbam"]
    if pd.isna(key_mlbam):
        raise ValueError(f"Could not resolve key_mlbam for player '{pitcher_name}'.")

    return int(key_mlbam)


def get_pitcher_statcast(
    pitcher_name: str,
    start_dt: str,
    end_dt: str,
    cache_dir: str = "data",
) -> pd.DataFrame:
    player_id = _lookup_player_id(pitcher_name)

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    cache_file = cache_path / f"statcast_{player_id}_{start_dt}_{end_dt}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    data = pybaseball.statcast_pitcher(start_dt, end_dt, player_id)
    if data is None or data.empty:
        df = pd.DataFrame()
    else:
        df = data.dropna(subset=REQUIRED_COLUMNS).copy()

    df.to_parquet(cache_file, index=False)
    return df
