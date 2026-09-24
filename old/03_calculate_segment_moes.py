import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely

import paths
from parallel_utils import (
    build_journey_partitions,
    reset_generated_directory,
    resolve_workers,
)
from progress import Progress


PARENTS = None
PARENT_INDEX = None
SEGMENT_COUNTS = None

OUTPUT_COLUMNS = [
    "moe_segment_id", "link_key", "segment_index", "time_bin",
    "parent_start_ft", "parent_end_ft", "segment_length_ft",
    "parent_length_ft", "is_parent_start", "is_parent_end",
    "posted_speed_limit_mph", "speed_match_status", "avg_speed_mph",
    "speed_reduction_pct", "waypoint_count", "journey_count",
    "slow_movement_count", "slow_movement_pct", "dsh_trip_count", "DSH",
    "complete_traversal_count", "avg_travel_time_s",
]

OUTPUT_SCHEMA = pa.schema([
    ("moe_segment_id", pa.string()),
    ("link_key", pa.string()),
    ("segment_index", pa.int64()),
    ("time_bin", pa.timestamp("ns")),
    ("parent_start_ft", pa.float64()),
    ("parent_end_ft", pa.float64()),
    ("segment_length_ft", pa.float64()),
    ("parent_length_ft", pa.float64()),
    ("is_parent_start", pa.bool_()),
    ("is_parent_end", pa.bool_()),
    ("posted_speed_limit_mph", pa.float64()),
    ("speed_match_status", pa.string()),
    ("avg_speed_mph", pa.float64()),
    ("speed_reduction_pct", pa.float64()),
    ("waypoint_count", pa.float64()),
    ("journey_count", pa.float64()),
    ("slow_movement_count", pa.float64()),
    ("slow_movement_pct", pa.float64()),
    ("dsh_trip_count", pa.float64()),
    ("DSH", pa.float64()),
    ("complete_traversal_count", pa.float64()),
    ("avg_travel_time_s", pa.float64()),
])


# Load parent geometry once inside each worker process.
def initialize_worker():
    global PARENTS, PARENT_INDEX, SEGMENT_COUNTS
    parents = pd.read_parquet(
        paths.PARENT_FILE,
        columns=["link_key", "parent_length_ft", "geometry_wkb"],
    )
    parents["link_key"] = parents["link_key"].astype(str)
    parents["geometry"] = shapely.from_wkb(
        parents.pop("geometry_wkb").to_numpy()
    )
    parents = parents.reset_index(drop=True)
    PARENTS = parents
    PARENT_INDEX = pd.Series(
        parents.index.to_numpy(), index=parents["link_key"]
    ).to_dict()
    counts = np.maximum(
        1,
        np.ceil(
            parents["parent_length_ft"].to_numpy(float)
            / paths.SEGMENT_LENGTH_FT - 1e-12
        ).astype(int),
    )
    SEGMENT_COUNTS = dict(zip(parents["link_key"], counts))


# Locate every CV observation along its parent and assign a 200-ft segment.
def assign_segments(data, parents=None, parent_index=None):
    parents = PARENTS if parents is None else parents
    parent_index = PARENT_INDEX if parent_index is None else parent_index
    data = data[data["journey_id"].notna() & data["link_key"].notna()].copy()
    data["link_key"] = data["link_key"].astype(str)
    indices = data["link_key"].map(parent_index)
    keep = indices.notna().to_numpy()
    data = data.loc[keep].copy()
    indices = indices.loc[keep].astype(int).to_numpy()
    if data.empty:
        return data

    x = pd.to_numeric(data["x_ft"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(data["y_ft"], errors="coerce").to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(y)
    data = data.loc[valid].copy()
    indices = indices[valid]
    x = x[valid]
    y = y[valid]
    if data.empty:
        return data

    points = shapely.points(x / paths.M_TO_FT, y / paths.M_TO_FT)
    lines = parents["geometry"].to_numpy()[indices]
    position_ft = shapely.line_locate_point(lines, points) * paths.M_TO_FT
    length_ft = parents["parent_length_ft"].to_numpy(float)[indices]
    position_ft = np.clip(position_ft, 0.0, np.nextafter(length_ft, -np.inf))
    segment_index = np.floor(position_ft / paths.SEGMENT_LENGTH_FT).astype(int) + 1

    data["parent_position_ft"] = position_ft
    data["parent_length_ft"] = length_ft
    data["segment_index"] = segment_index
    data["moe_segment_id"] = (
        data["link_key"] + "__"
        + pd.Series(segment_index, index=data.index).astype(str).str.zfill(4)
    )
    data["capture_time"] = pd.to_numeric(data["capture_time"], errors="coerce")
    data["local_time"] = pd.to_datetime(data["local_time"], errors="coerce")
    data["speed_mph"] = pd.to_numeric(data["speed_mph"], errors="coerce")
    return data.dropna(subset=["capture_time", "local_time"])


# Split ordered journeys after long gaps and identify parent and segment runs.
def prepare_sequence(data):
    if data.empty:
        return data
    data = data.sort_values(["journey_id", "capture_time"]).reset_index(drop=True)
    new_journey = data["journey_id"].ne(
        data["journey_id"].shift()
    ).fillna(True).to_numpy(dtype=bool)
    gap = data.groupby("journey_id", sort=False)["capture_time"].diff()
    long_gap = gap.gt(
        paths.MAX_WAYPOINT_GAP_SECONDS
    ).fillna(False).to_numpy(dtype=bool)
    new_trip = new_journey | long_gap
    trip_number = np.cumsum(new_trip)
    data["trip_id"] = (
        data["journey_id"].astype(str) + "::" + trip_number.astype(str)
    )
    parent_change = data["link_key"].ne(
        data["link_key"].shift()
    ).fillna(True).to_numpy(dtype=bool)
    segment_change = data["moe_segment_id"].ne(
        data["moe_segment_id"].shift()
    ).fillna(True).to_numpy(dtype=bool)
    data["parent_run_id"] = np.cumsum(new_trip | parent_change)
    data["segment_run_id"] = np.cumsum(new_trip | segment_change)
    data["time_bin"] = data["local_time"].dt.floor(
        f"{paths.TIME_BIN_MINUTES}min"
    )
    return data


# Calculate speed and DSH components for complete journey partitions.
def calculate_speed_components(data):
    valid = data[
        data["speed_mph"].notna()
        & np.isfinite(data["speed_mph"])
        & data["speed_mph"].ge(0)
    ].copy()
    if valid.empty:
        return None, None

    valid["is_slow"] = valid["speed_mph"].lt(paths.SLOW_SPEED_THRESHOLD_MPH)
    valid["slow_speed"] = np.where(valid["is_slow"], valid["speed_mph"], 0.0)
    basic = valid.groupby(["moe_segment_id", "time_bin"], as_index=False).agg(
        waypoint_count=("speed_mph", "size"),
        journey_count=("journey_id", "nunique"),
        total_speed_sum=("speed_mph", "sum"),
        slow_movement_count=("is_slow", "sum"),
        slow_speed_sum=("slow_speed", "sum"),
    )

    valid["speed_change"] = valid.groupby(
        ["segment_run_id", "time_bin"], sort=False
    )["speed_mph"].diff().abs()
    run_dsh = valid.groupby(
        ["trip_id", "moe_segment_id", "segment_run_id", "time_bin"],
        as_index=False,
    ).agg(
        waypoint_count=("speed_mph", "size"),
        speed_change_sum=("speed_change", "sum"),
    )
    run_dsh = run_dsh[run_dsh["waypoint_count"] >= 2].copy()
    if run_dsh.empty:
        return basic, None
    run_dsh["trip_dsh"] = run_dsh["speed_change_sum"] / run_dsh["waypoint_count"]
    dsh = run_dsh.groupby(["moe_segment_id", "time_bin"], as_index=False).agg(
        dsh_sum=("trip_dsh", "sum"),
        dsh_trip_count=("trip_dsh", "size"),
    )
    return basic, dsh


# Allocate complete parent traversal time to its ordered 200-ft segments.
def calculate_travel_components(data, segment_counts=None):
    segment_counts = SEGMENT_COUNTS if segment_counts is None else segment_counts
    if data.empty:
        return None
    grouped = list(data.groupby("parent_run_id", sort=False))
    if len(grouped) < 3:
        return None

    records = []
    for position in range(1, len(grouped) - 1):
        _, run = grouped[position]
        _, previous = grouped[position - 1]
        _, following = grouped[position + 1]
        trip_id = run["trip_id"].iloc[0]
        if (
            previous["trip_id"].iloc[0] != trip_id
            or following["trip_id"].iloc[0] != trip_id
        ):
            continue

        link_key = str(run["link_key"].iloc[0])
        if link_key not in segment_counts:
            continue
        entry_time = float(run["capture_time"].iloc[0])
        exit_time = float(following["capture_time"].iloc[0])
        if not np.isfinite(entry_time) or not np.isfinite(exit_time) or exit_time <= entry_time:
            continue

        parent_length = float(run["parent_length_ft"].iloc[0])
        ordered = run.sort_values("capture_time")
        positions = np.maximum.accumulate(
            np.clip(ordered["parent_position_ft"].to_numpy(float), 0.0, parent_length)
        )
        times = ordered["capture_time"].to_numpy(float)
        anchor_frame = pd.DataFrame({
            "position": np.concatenate(([0.0], positions, [parent_length])),
            "time": np.concatenate(([entry_time], times, [exit_time])),
        })
        anchors = anchor_frame.groupby("position", sort=True)["time"].max()
        anchors.loc[0.0] = entry_time
        anchors = anchors.sort_index()
        if len(anchors) < 2:
            continue

        count = int(segment_counts[link_key])
        boundaries = np.minimum(
            np.arange(count + 1, dtype=float) * paths.SEGMENT_LENGTH_FT,
            parent_length,
        )
        boundary_times = np.interp(
            boundaries, anchors.index.to_numpy(float), anchors.to_numpy(float)
        )
        travel_times = np.diff(boundary_times)
        local_starts = ordered["local_time"].iloc[0] + pd.to_timedelta(
            boundary_times[:-1] - entry_time, unit="s"
        )
        for index, travel_time in enumerate(travel_times, start=1):
            if np.isfinite(travel_time) and travel_time > 0:
                records.append({
                    "moe_segment_id": f"{link_key}__{index:04d}",
                    "time_bin": local_starts[index - 1].floor(
                        f"{paths.TIME_BIN_MINUTES}min"
                    ),
                    "travel_time_sum": travel_time,
                    "complete_traversal_count": 1,
                })

    if not records:
        return None
    return pd.DataFrame.from_records(records).groupby(
        ["moe_segment_id", "time_bin"], as_index=False
    ).agg(
        travel_time_sum=("travel_time_sum", "sum"),
        complete_traversal_count=("complete_traversal_count", "sum"),
    )


# Process one complete journey partition and write compact aggregate components.
def process_partition(partition_file, partial_directory):
    partition_file = Path(partition_file)
    partial_directory = Path(partial_directory)
    number = int(partition_file.stem.rsplit("_", 1)[-1])
    data = pd.read_parquet(partition_file)
    assigned = assign_segments(data)
    sequence = prepare_sequence(assigned)
    basic, dsh = calculate_speed_components(sequence)
    travel = calculate_travel_components(sequence)

    files = {}
    for label, frame in (("basic", basic), ("dsh", dsh), ("travel", travel)):
        if frame is not None and not frame.empty:
            file = partial_directory / f"{label}_{number:03d}.parquet"
            frame.to_parquet(
                file, index=False, compression=paths.PARQUET_COMPRESSION
            )
            files[label] = str(file)
    return {
        "source_rows": len(data),
        "assigned_rows": len(assigned),
        "files": files,
    }


# Reduce worker aggregates after each journey has been processed exactly once.
def reduce_partials(partial_directory):
    def load(label):
        files = sorted(partial_directory.glob(f"{label}_*.parquet"))
        return [pd.read_parquet(file) for file in files]

    basic_parts = load("basic")
    if not basic_parts:
        raise RuntimeError("No valid CV speed observations were assigned to segments.")
    basic = pd.concat(basic_parts, ignore_index=True).groupby(
        ["moe_segment_id", "time_bin"], as_index=False
    ).agg(
        waypoint_count=("waypoint_count", "sum"),
        journey_count=("journey_count", "sum"),
        total_speed_sum=("total_speed_sum", "sum"),
        slow_movement_count=("slow_movement_count", "sum"),
        slow_speed_sum=("slow_speed_sum", "sum"),
    )
    basic["avg_speed_mph"] = basic["total_speed_sum"] / basic["waypoint_count"]
    basic["slow_movement_pct"] = np.where(
        basic["total_speed_sum"] > 0,
        100 * basic["slow_speed_sum"] / basic["total_speed_sum"],
        np.nan,
    )

    dsh_parts = load("dsh")
    if dsh_parts:
        dsh = pd.concat(dsh_parts, ignore_index=True).groupby(
            ["moe_segment_id", "time_bin"], as_index=False
        ).agg(dsh_sum=("dsh_sum", "sum"), dsh_trip_count=("dsh_trip_count", "sum"))
        dsh["DSH"] = dsh["dsh_sum"] / dsh["dsh_trip_count"]
        basic = basic.merge(
            dsh.drop(columns="dsh_sum"),
            on=["moe_segment_id", "time_bin"], how="left",
        )

    travel_parts = load("travel")
    if travel_parts:
        travel = pd.concat(travel_parts, ignore_index=True).groupby(
            ["moe_segment_id", "time_bin"], as_index=False
        ).agg(
            travel_time_sum=("travel_time_sum", "sum"),
            complete_traversal_count=("complete_traversal_count", "sum"),
        )
        travel["avg_travel_time_s"] = (
            travel["travel_time_sum"] / travel["complete_traversal_count"]
        )
        basic = basic.merge(
            travel.drop(columns="travel_time_sum"),
            on=["moe_segment_id", "time_bin"], how="left",
        )

    for column in ["dsh_trip_count", "DSH", "complete_traversal_count", "avg_travel_time_s"]:
        if column not in basic:
            basic[column] = np.nan
    return basic


# Write the full segment-time inventory in bounded batches instead of one huge frame.
def write_final_output(observed, segments):
    time_bins = pd.date_range(
        observed["time_bin"].min(), observed["time_bin"].max(),
        freq=f"{paths.TIME_BIN_MINUTES}min",
    )
    metric_columns = [
        column for column in observed.columns
        if column not in {"moe_segment_id", "time_bin"}
    ]
    observed = observed.set_index(["moe_segment_id", "time_bin"]).sort_index()
    metadata = segments.drop(columns="geometry_wkb")
    temporary = paths.MOE_FILE.with_name(paths.MOE_FILE.stem + ".partial.parquet")
    if temporary.exists():
        temporary.unlink()
    progress = Progress("Final MOE writing", paths.REPORT_INTERVAL_SECONDS)

    with pq.ParquetWriter(
        temporary, OUTPUT_SCHEMA, compression=paths.PARQUET_COMPRESSION
    ) as writer:
        for start in range(0, len(metadata), paths.FINAL_SEGMENT_BATCH_SIZE):
            segment_batch = metadata.iloc[
                start:start + paths.FINAL_SEGMENT_BATCH_SIZE
            ].copy()
            grid = segment_batch.merge(
                pd.DataFrame({"time_bin": time_bins}), how="cross"
            )
            keys = pd.MultiIndex.from_frame(grid[["moe_segment_id", "time_bin"]])
            values = observed.reindex(keys).reset_index(drop=True)
            result = pd.concat(
                [grid.reset_index(drop=True), values[metric_columns]], axis=1
            )
            result["speed_reduction_pct"] = np.where(
                result["posted_speed_limit_mph"] > 0,
                (result["posted_speed_limit_mph"] - result["avg_speed_mph"])
                / result["posted_speed_limit_mph"] * 100,
                np.nan,
            )
            for column in OUTPUT_COLUMNS:
                if column not in result:
                    result[column] = np.nan
            numeric = [
                "avg_speed_mph", "speed_reduction_pct", "slow_movement_pct",
                "DSH", "avg_travel_time_s",
            ]
            result[numeric] = result[numeric].round(3)
            table = pa.Table.from_pandas(
                result[OUTPUT_COLUMNS], schema=OUTPUT_SCHEMA,
                preserve_index=False, safe=False,
            )
            writer.write_table(table)
            progress.report(
                min(start + len(segment_batch), len(metadata)), len(metadata)
            )

    temporary.replace(paths.MOE_FILE)
    progress.report(len(metadata), len(metadata), force=True)
    return len(metadata) * len(time_bins), len(time_bins)


# Calculate network-wide MOEs with journey-safe process parallelism.
def calculate_moes(workers=None, rebuild_partitions=False, max_cv_rows=None):
    workers = resolve_workers(workers)
    partition_files, partition_rows = build_journey_partitions(
        rebuild=rebuild_partitions, max_cv_rows=max_cv_rows
    )
    reset_generated_directory(paths.MOE_PARTIAL_DIR)
    assigned_rows = 0
    completed_rows = 0
    progress = Progress(
        "Parallel MOE calculation",
        paths.REPORT_INTERVAL_SECONDS,
        include_children=True,
    )

    print(f"Processing {len(partition_files)} journey partitions with {workers} workers")
    with ProcessPoolExecutor(
        max_workers=workers, initializer=initialize_worker
    ) as executor:
        futures = {
            executor.submit(
                process_partition, str(file), str(paths.MOE_PARTIAL_DIR)
            ): file
            for file in partition_files
        }
        for future in as_completed(futures):
            result = future.result()
            completed_rows += result["source_rows"]
            assigned_rows += result["assigned_rows"]
            progress.report(completed_rows, partition_rows, force=True)

    print("Reducing worker aggregates...")
    observed = reduce_partials(paths.MOE_PARTIAL_DIR)
    segments = pd.read_parquet(paths.SEGMENT_FILE)
    segment_time_rows, time_bin_count = write_final_output(observed, segments)
    print("\nFIVE-MINUTE MOE CALCULATION COMPLETE")
    print(f"Workers: {workers}")
    print(f"CV rows partitioned: {partition_rows:,}")
    print(f"Observations assigned: {assigned_rows:,}")
    print(f"Segments: {len(segments):,}")
    print(f"Time bins: {time_bin_count:,}")
    print(f"Segment-time rows: {segment_time_rows:,}")
    print("Saved:", paths.MOE_FILE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int)
    parser.add_argument("--rebuild-partitions", action="store_true")
    parser.add_argument("--max-cv-rows", type=int)
    args = parser.parse_args()
    calculate_moes(
        workers=args.workers,
        rebuild_partitions=args.rebuild_partitions,
        max_cv_rows=args.max_cv_rows,
    )


if __name__ == "__main__":
    paths.check_inputs()
    main()
