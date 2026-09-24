import numpy as np
import pandas as pd

from moes.common import sum_partials, write_measure


TABLE_NAME = "link_ave_speed"
_COLUMNS = ["speed_sum", "sample_size"]


def partial(sequence):
    valid = sequence[
        sequence["speed_mph"].notna()
        & np.isfinite(sequence["speed_mph"])
        & sequence["speed_mph"].ge(0)
    ]
    if valid.empty:
        return None
    grouped = valid.groupby(["moe_segment_id", "time_bin"], as_index=False).agg(
        speed_sum=("speed_mph", "sum"),
        sample_size=("speed_mph", "size"),
    )
    return grouped


def reduce(running, new):
    return sum_partials(running, new, _COLUMNS)


def values_from(totals, segments):
    if totals is None or totals.empty:
        return pd.DataFrame(columns=["moe_segment_id", "time_bin", "value", "sample_size"])
    totals = totals.copy()
    totals["value"] = totals["speed_sum"] / totals["sample_size"]
    return totals[["moe_segment_id", "time_bin", "value", "sample_size"]]


def write(totals, segments):
    write_measure(TABLE_NAME, segments, values_from(totals, segments))


def from_attributes(attributes, segments):
    if attributes is None or attributes.empty:
        return pd.DataFrame(columns=["moe_segment_id", "time_bin", "value", "sample_size"])
    frame = attributes.rename(columns={"speed_count": "sample_size"})
    return values_from(frame, segments)


def generate(sequence, segments):
    write(partial(sequence), segments)
