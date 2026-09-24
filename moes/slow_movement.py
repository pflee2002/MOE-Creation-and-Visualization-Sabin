import numpy as np
import pandas as pd

from moes.common import env_int, sum_partials, write_measure


TABLE_NAME = "link_slow_movement"
_COLUMNS = ["total_speed_sum", "slow_speed_sum", "sample_size"]


def partial(sequence):
    valid = sequence[
        sequence["speed_mph"].notna()
        & np.isfinite(sequence["speed_mph"])
        & sequence["speed_mph"].ge(0)
    ].copy()
    if valid.empty:
        return None
    valid["is_slow"] = valid["speed_mph"].lt(env_int("SLOW_SPEED_THRESHOLD_MPH", 5))
    valid["slow_speed"] = np.where(valid["is_slow"], valid["speed_mph"], 0.0)
    return valid.groupby(["moe_segment_id", "time_bin"], as_index=False).agg(
        total_speed_sum=("speed_mph", "sum"),
        slow_speed_sum=("slow_speed", "sum"),
        sample_size=("is_slow", "sum"),
    )


def reduce(running, new):
    return sum_partials(running, new, _COLUMNS)


def values_from(totals, segments):
    if totals is None or totals.empty:
        return pd.DataFrame(columns=["moe_segment_id", "time_bin", "value", "sample_size"])
    totals = totals.copy()
    totals["value"] = np.where(
        totals["total_speed_sum"] > 0,
        100 * totals["slow_speed_sum"] / totals["total_speed_sum"],
        np.nan,
    )
    return totals[["moe_segment_id", "time_bin", "value", "sample_size"]]


def write(totals, segments):
    write_measure(TABLE_NAME, segments, values_from(totals, segments))


def from_attributes(attributes, segments):
    if attributes is None or attributes.empty:
        return pd.DataFrame(columns=["moe_segment_id", "time_bin", "value", "sample_size"])
    frame = attributes.rename(columns={
        "speed_sum": "total_speed_sum",
        "slow_count": "sample_size",
    })
    return values_from(frame, segments)


def generate(sequence, segments):
    write(partial(sequence), segments)
