import numpy as np
import pandas as pd

from moes.common import sum_partials, write_measure


TABLE_NAME = "link_dsh"
_COLUMNS = ["dsh_sum", "sample_size"]


def partial(sequence):
    valid = sequence[
        sequence["speed_mph"].notna()
        & np.isfinite(sequence["speed_mph"])
        & sequence["speed_mph"].ge(0)
    ].copy()
    if valid.empty:
        return None
    valid["speed_change"] = valid.groupby(
        ["segment_run_id", "time_bin"], sort=False
    )["speed_mph"].diff().abs()
    runs = valid.groupby(
        ["trip_id", "moe_segment_id", "segment_run_id", "time_bin"],
        as_index=False,
    ).agg(
        waypoint_count=("speed_mph", "size"),
        speed_change_sum=("speed_change", "sum"),
    )
    runs = runs[runs["waypoint_count"] >= 2]
    if runs.empty:
        return None
    runs = runs.copy()
    runs["trip_dsh"] = runs["speed_change_sum"] / runs["waypoint_count"]
    return runs.groupby(["moe_segment_id", "time_bin"], as_index=False).agg(
        dsh_sum=("trip_dsh", "sum"),
        sample_size=("trip_dsh", "size"),
    )


def reduce(running, new):
    return sum_partials(running, new, _COLUMNS)


def values_from(totals, segments):
    if totals is None or totals.empty:
        return pd.DataFrame(columns=["moe_segment_id", "time_bin", "value", "sample_size"])
    totals = totals.copy()
    totals["value"] = totals["dsh_sum"] / totals["sample_size"]
    return totals[["moe_segment_id", "time_bin", "value", "sample_size"]]


def write(totals, segments):
    write_measure(TABLE_NAME, segments, values_from(totals, segments))


def generate(sequence, segments):
    write(partial(sequence), segments)
