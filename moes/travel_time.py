import numpy as np
import pandas as pd

from moes.common import pack_link_value, write_packed


TABLE_NAME = "link_travel_time"
SINGLE_VALUE = True
_FEET_PER_MILE = 5280.0
_SECONDS_PER_HOUR = 3600.0


def piece_times(attributes, segments):
    columns = ["link_key", "time_bin", "segment_length_ft", "piece_seconds", "speed_count"]
    if attributes is None or attributes.empty:
        return pd.DataFrame(columns=columns)
    pieces = segments[
        ["moe_segment_id", "link_key", "segment_length_ft", "posted_speed_limit_mph"]
    ].drop_duplicates("moe_segment_id")
    pieces["link_key"] = pieces["link_key"].astype(str)
    observed = attributes.merge(
        pieces[["moe_segment_id", "link_key"]], on="moe_segment_id", how="inner"
    )
    if observed.empty:
        return pd.DataFrame(columns=columns)
    grid = observed[["link_key", "time_bin"]].drop_duplicates().merge(
        pieces, on="link_key", how="left"
    )
    grid = grid.merge(
        attributes[["moe_segment_id", "time_bin", "speed_sum", "speed_count"]],
        on=["moe_segment_id", "time_bin"],
        how="left",
    )
    grid["speed_count"] = grid["speed_count"].fillna(0)
    grid["speed_sum"] = grid["speed_sum"].fillna(0)
    has_sample = grid["speed_count"].gt(0).to_numpy()
    observed_speed = grid["speed_sum"].to_numpy(float) / np.where(
        has_sample, grid["speed_count"].to_numpy(float), 1.0
    )
    free_speed = grid["posted_speed_limit_mph"].to_numpy(float)
    speed = np.where(has_sample, observed_speed, free_speed)
    usable = np.isfinite(speed) & (speed > 0)
    grid = grid.loc[usable].copy()
    if grid.empty:
        return pd.DataFrame(columns=columns)
    grid["piece_seconds"] = (
        grid["segment_length_ft"].to_numpy(float) * _SECONDS_PER_HOUR
        / (_FEET_PER_MILE * speed[usable])
    )
    return grid[columns]


def from_attributes(attributes, segments):
    pieces = piece_times(attributes, segments)
    if pieces.empty:
        return pd.DataFrame(columns=["link_key", "time_bin", "value", "sample_size"])
    return pieces.groupby(["link_key", "time_bin"], as_index=False).agg(
        value=("piece_seconds", "sum"),
        sample_size=("speed_count", "sum"),
    )


def write(attributes, segments):
    write_packed(TABLE_NAME, pack_link_value(from_attributes(attributes, segments)))
