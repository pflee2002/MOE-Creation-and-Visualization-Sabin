import os
import re
import tempfile

import pandas as pd
import psycopg2

COLUMNS = (
    "from_time",
    "to_time",
    "from_local_time",
    "to_local_time",
    "link_key",
    "moe",
    "sample_size",
)


def env_int(name, default):
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def connect():
    return psycopg2.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ["DB_PORT"]),
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        dbname=os.environ["CV_DATABASE"],
    )


def _format_value(value):
    if pd.isna(value):
        return "-1"
    number = float(value)
    if number == -1:
        return "-1"
    text = f"{number:.3f}".rstrip("0").rstrip(".")
    return text if text not in {"", "-"} else "0"


def pack_link_rows(segments, segment_values):
    pieces = segments[["link_key", "segment_index", "moe_segment_id"]].drop_duplicates(
        "moe_segment_id"
    )
    pieces["link_key"] = pieces["link_key"].astype(str)
    if segment_values is None or segment_values.empty:
        return pd.DataFrame(columns=list(COLUMNS))

    values = segment_values.copy()
    values["time_bin"] = pd.to_datetime(values["time_bin"])
    observed = values.merge(
        pieces[["moe_segment_id", "link_key"]],
        on="moe_segment_id",
        how="inner",
    )
    pairs = observed[["link_key", "time_bin"]].drop_duplicates()
    grid = pairs.merge(pieces, on="link_key", how="left")
    grid = grid.merge(
        values[["moe_segment_id", "time_bin", "value", "sample_size"]],
        on=["moe_segment_id", "time_bin"],
        how="left",
    )
    grid["value_text"] = grid["value"].map(_format_value)
    grid["sample_text"] = grid["sample_size"].fillna(0).astype(int).astype(str)
    grid = grid.sort_values(["time_bin", "link_key", "segment_index"])
    packed = grid.groupby(["time_bin", "link_key"], sort=False).agg(
        moe=("value_text", ";".join),
        sample_size=("sample_text", ";".join),
    ).reset_index()
    packed["from_local_time"] = packed["time_bin"]
    packed["to_local_time"] = packed["time_bin"] + pd.to_timedelta(
        env_int("TIME_BIN_MINUTES", 15), unit="min"
    )
    local = pd.DatetimeIndex(packed["from_local_time"]).tz_localize("America/Chicago")
    packed["from_time"] = local.as_unit("ns").tz_convert("UTC").astype("int64") // 1_000_000_000
    packed["to_time"] = packed["from_time"] + env_int("TIME_BIN_MINUTES", 15) * 60
    return packed[list(COLUMNS)]


def pack_link_value(link_values):
    if link_values is None or link_values.empty:
        return pd.DataFrame(columns=list(COLUMNS))
    packed = link_values.copy()
    packed["time_bin"] = pd.to_datetime(packed["time_bin"])
    packed["moe"] = packed["value"].map(_format_value)
    packed["sample_size"] = packed["sample_size"].fillna(0).astype(int).astype(str)
    packed["from_local_time"] = packed["time_bin"]
    packed["to_local_time"] = packed["time_bin"] + pd.to_timedelta(
        env_int("TIME_BIN_MINUTES", 15), unit="min"
    )
    local = pd.DatetimeIndex(packed["from_local_time"]).tz_localize("America/Chicago")
    packed["from_time"] = local.as_unit("ns").tz_convert("UTC").astype("int64") // 1_000_000_000
    packed["to_time"] = packed["from_time"] + env_int("TIME_BIN_MINUTES", 15) * 60
    return packed[list(COLUMNS)]


def sum_partials(running, new, columns):
    frames = [
        frame for frame in (running, new)
        if frame is not None and not frame.empty
    ]
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True).groupby(
        ["moe_segment_id", "time_bin"], as_index=False
    )[columns].sum()


def _frame_list(rows):
    if rows is None:
        return []
    if isinstance(rows, pd.DataFrame):
        return [] if rows.empty else [rows]
    return [frame for frame in rows if frame is not None and not frame.empty]


def copy_frames(cursor, table_name, columns, frames):
    column_list = ", ".join(columns)
    total = 0
    while frames:
        frame = frames.pop()
        part = frame.loc[:, list(columns)]
        del frame
        total += len(part)
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            part.to_csv(path, index=False, header=False, lineterminator="\n")
            del part
            with open(path, "r", encoding="utf-8", newline="") as handle:
                cursor.copy_expert(
                    f"COPY {table_name} ({column_list}) FROM STDIN WITH CSV",
                    handle,
                )
        finally:
            os.remove(path)
    return total


def write_packed(table_name, rows):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
        raise ValueError(f"Unsafe SQL identifier: {table_name}")
    frames = _frame_list(rows)
    connection = connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    from_time bigint NOT NULL,
                    to_time bigint NOT NULL,
                    from_local_time timestamp NOT NULL,
                    to_local_time timestamp NOT NULL,
                    link_key text NOT NULL,
                    moe text NOT NULL,
                    sample_size text NOT NULL,
                    PRIMARY KEY (from_time, link_key)
                )
                """
            )
            cursor.execute(f"TRUNCATE TABLE {table_name}")
            count = copy_frames(cursor, table_name, COLUMNS, frames)
        connection.commit()
    finally:
        connection.close()
    print(f"{table_name}: {count:,} link-time rows")


def write_measure(table_name, segments, segment_values):
    write_packed(table_name, pack_link_rows(segments, segment_values))
