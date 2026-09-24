import pandas as pd

from moes.common import connect, copy_frames, env_int, _frame_list
from moes.travel_time import _FEET_PER_MILE, _SECONDS_PER_HOUR, piece_times


TABLE_NAME = "link_speed"
COLUMNS = (
    "from_time",
    "to_time",
    "from_local_time",
    "to_local_time",
    "link_key",
    "link_length_ft",
    "travel_time_sec",
    "measured_speed_mph",
)


def from_attributes(attributes, segments):
    pieces = piece_times(attributes, segments)
    if pieces.empty:
        return pd.DataFrame(columns=["link_key", "time_bin", *COLUMNS[5:]])
    totals = pieces.groupby(["link_key", "time_bin"], as_index=False).agg(
        link_length_ft=("segment_length_ft", "sum"),
        travel_time_sec=("piece_seconds", "sum"),
    )
    totals = totals.loc[totals["travel_time_sec"] > 0].copy()
    if totals.empty:
        return pd.DataFrame(columns=["link_key", "time_bin", *COLUMNS[5:]])
    totals["measured_speed_mph"] = (
        totals["link_length_ft"] * _SECONDS_PER_HOUR
        / (_FEET_PER_MILE * totals["travel_time_sec"])
    )
    return totals


def pack(link_values):
    if link_values is None or link_values.empty:
        return pd.DataFrame(columns=list(COLUMNS))
    packed = link_values.copy()
    packed["time_bin"] = pd.to_datetime(packed["time_bin"])
    packed["from_local_time"] = packed["time_bin"]
    packed["to_local_time"] = packed["time_bin"] + pd.to_timedelta(
        env_int("TIME_BIN_MINUTES", 15), unit="min"
    )
    local = pd.DatetimeIndex(packed["from_local_time"]).tz_localize("America/Chicago")
    packed["from_time"] = local.as_unit("ns").tz_convert("UTC").astype("int64") // 1_000_000_000
    packed["to_time"] = packed["from_time"] + env_int("TIME_BIN_MINUTES", 15) * 60
    return packed[list(COLUMNS)]


def write_rows(rows):
    frames = _frame_list(rows)
    connection = connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                    from_time bigint NOT NULL,
                    to_time bigint NOT NULL,
                    from_local_time timestamp NOT NULL,
                    to_local_time timestamp NOT NULL,
                    link_key text NOT NULL,
                    link_length_ft double precision NOT NULL,
                    travel_time_sec double precision NOT NULL,
                    measured_speed_mph double precision NOT NULL,
                    PRIMARY KEY (from_time, link_key)
                )
                """
            )
            cursor.execute(f"TRUNCATE TABLE {TABLE_NAME}")
            count = copy_frames(cursor, TABLE_NAME, COLUMNS, frames)
        connection.commit()
    finally:
        connection.close()
    print(f"{TABLE_NAME}: {count:,} link-time rows")
