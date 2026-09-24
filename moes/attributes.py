import numpy as np
import pandas as pd

from moes.common import env_int, sum_partials


# Columns stored on each link segment and five-minute bin.
# Add a column here when a new measure needs another observation attribute.
COLUMNS = ["speed_sum", "speed_count", "slow_speed_sum", "slow_count"]


def add(totals, assigned):
    if assigned is None or assigned.empty:
        return totals
    speed = pd.to_numeric(assigned["speed_mph"], errors="coerce")
    valid = speed.notna() & np.isfinite(speed) & speed.ge(0)
    if not valid.any():
        return totals
    frame = assigned.loc[valid, ["moe_segment_id", "time_bin"]].copy()
    frame["speed_mph"] = speed.loc[valid].to_numpy(float)
    slow = frame["speed_mph"] < env_int("SLOW_SPEED_THRESHOLD_MPH", 5)
    frame["slow_speed"] = np.where(slow, frame["speed_mph"], 0.0)
    frame["is_slow"] = slow
    grouped = frame.groupby(["moe_segment_id", "time_bin"], as_index=False).agg(
        speed_sum=("speed_mph", "sum"),
        speed_count=("speed_mph", "size"),
        slow_speed_sum=("slow_speed", "sum"),
        slow_count=("is_slow", "sum"),
    )
    return sum_partials(totals, grouped, COLUMNS)
