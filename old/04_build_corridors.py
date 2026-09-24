import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd

import paths
from parallel_utils import (
    build_journey_partitions,
    reset_generated_directory,
    resolve_workers,
)
from progress import Progress


PARENT_LINKS = None


# Normalize node identifiers so topology joins remain stable.
def node_key(value):
    if pd.isna(value):
        return None
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value).strip()


# Load valid parent keys once inside every transition-count worker.
def initialize_transition_worker():
    global PARENT_LINKS
    parent = pd.read_parquet(paths.PARENT_FILE, columns=["link_key"])
    PARENT_LINKS = set(parent["link_key"].astype(str))


# Convert complete journeys into distinct directed parent-link transitions.
def transitions_from_rows(data, parent_links=None):
    parent_links = PARENT_LINKS if parent_links is None else parent_links
    if data.empty:
        return None
    data = data.copy()
    data["journey_id"] = data["journey_id"].astype(str)
    data["link_key"] = data["link_key"].astype("string")
    data["capture_time"] = pd.to_numeric(data["capture_time"], errors="coerce")
    data = data.dropna(subset=["capture_time"])
    if data.empty:
        return None
    data = data.sort_values(["journey_id", "capture_time"]).reset_index(drop=True)
    new_journey = data["journey_id"].ne(
        data["journey_id"].shift()
    ).fillna(True).to_numpy(dtype=bool)
    gap = data.groupby("journey_id", sort=False)["capture_time"].diff()
    long_gap = gap.gt(
        paths.MAX_WAYPOINT_GAP_SECONDS
    ).fillna(False).to_numpy(dtype=bool)
    link = data["link_key"].where(data["link_key"].isin(parent_links))
    link_change = link.fillna("__missing__").ne(
        link.fillna("__missing__").shift()
    ).fillna(True).to_numpy(dtype=bool)
    run_id = np.cumsum(new_journey | long_gap | link_change)
    runs = data.assign(link_key=link, run_id=run_id).groupby(
        "run_id", as_index=False, sort=False
    ).agg(
        journey_id=("journey_id", "first"),
        link_key=("link_key", "first"),
    )
    next_journey = runs["journey_id"].shift(-1)
    next_link = runs["link_key"].shift(-1)
    valid = (
        runs["journey_id"].eq(next_journey)
        & runs["link_key"].notna()
        & next_link.notna()
        & runs["link_key"].ne(next_link)
    )
    if not valid.any():
        return None
    result = pd.DataFrame({
        "link_key": runs.loc[valid, "link_key"].astype(str).to_numpy(),
        "next_link": next_link.loc[valid].astype(str).to_numpy(),
    })
    return result.groupby(["link_key", "next_link"], as_index=False).size().rename(
        columns={"size": "transition_count"}
    )


# Count transitions in one complete journey partition.
def process_transition_partition(partition_file, partial_directory):
    partition_file = Path(partition_file)
    partial_directory = Path(partial_directory)
    number = int(partition_file.stem.rsplit("_", 1)[-1])
    data = pd.read_parquet(
        partition_file, columns=["journey_id", "capture_time", "link_key"]
    )
    transitions = transitions_from_rows(data)
    output = None
    if transitions is not None and not transitions.empty:
        output = partial_directory / f"transitions_{number:03d}.parquet"
        transitions.to_parquet(
            output, index=False, compression=paths.PARQUET_COMPRESSION
        )
    return len(data), str(output) if output else None


# Count observed parent-link movements concurrently across journey partitions.
def build_transition_counts(workers=None, rebuild_partitions=False):
    workers = resolve_workers(workers)
    partition_files, partition_rows = build_journey_partitions(
        rebuild=rebuild_partitions
    )
    reset_generated_directory(paths.TRANSITION_PARTIAL_DIR)
    completed = 0
    progress = Progress(
        "Parallel transition counting",
        paths.REPORT_INTERVAL_SECONDS,
        include_children=True,
    )
    print(f"Counting transitions with {workers} workers")

    with ProcessPoolExecutor(
        max_workers=workers, initializer=initialize_transition_worker
    ) as executor:
        futures = {
            executor.submit(
                process_transition_partition,
                str(file),
                str(paths.TRANSITION_PARTIAL_DIR),
            ): file
            for file in partition_files
        }
        for future in as_completed(futures):
            rows, _ = future.result()
            completed += rows
            progress.report(completed, partition_rows, force=True)

    files = sorted(paths.TRANSITION_PARTIAL_DIR.glob("transitions_*.parquet"))
    if not files:
        return pd.DataFrame(columns=["link_key", "next_link", "transition_count"])
    return pd.concat(
        [pd.read_parquet(file) for file in files], ignore_index=True
    ).groupby(["link_key", "next_link"], as_index=False)["transition_count"].sum()


# Return the smallest circular difference between two directions.
def heading_difference(a, b):
    if pd.isna(a) or pd.isna(b):
        return np.inf
    difference = abs(float(a) - float(b))
    return min(difference, 360.0 - difference)


# Trace one connected path using topology, transition support, and heading.
def trace_corridor(start_link, by_link, outgoing, transition_lookup, unavailable):
    path = [start_link]
    methods = ["start"]
    supports = [np.nan]
    visited = {start_link}
    length_ft = float(by_link.loc[start_link, "parent_length_ft"])

    while (
        len(path) < paths.MAX_CORRIDOR_PARENT_LINKS
        and length_ft < paths.MAX_CORRIDOR_LENGTH_FT
    ):
        current = by_link.loc[path[-1]]
        candidates = [
            candidate for candidate in outgoing.get(current["to_node_key"], [])
            if candidate not in visited and candidate not in unavailable
        ]
        if not candidates:
            break
        non_reverse = [
            candidate for candidate in candidates
            if by_link.loc[candidate, "to_node_key"] != current["from_node_key"]
        ]
        if non_reverse:
            candidates = non_reverse

        ranked = []
        for candidate in candidates:
            support = int(transition_lookup.get((path[-1], candidate), 0))
            angle = heading_difference(
                current["heading"], by_link.loc[candidate, "heading"]
            )
            ranked.append((-support, angle, candidate, support))
        ranked.sort()
        _, _, selected, support = ranked[0]
        new_length = length_ft + float(by_link.loc[selected, "parent_length_ft"])
        if new_length > paths.MAX_CORRIDOR_LENGTH_FT and len(path) > 1:
            break
        path.append(selected)
        supports.append(support)
        methods.append("observed_transition" if support > 0 else "topology_heading")
        visited.add(selected)
        length_ft = new_length

    return path, supports, methods


# Build connected visualization corridors after network-wide MOEs are complete.
def build_corridors(workers=None, rebuild_partitions=False):
    parent_columns = [
        "link_key", "from_node_id", "to_node_id", "heading", "parent_length_ft",
        "facility_type", "posted_speed_limit_mph",
    ]
    parents = pd.read_parquet(paths.PARENT_FILE, columns=parent_columns)
    parents["link_key"] = parents["link_key"].astype(str)
    parents["from_node_key"] = parents["from_node_id"].map(node_key)
    parents["to_node_key"] = parents["to_node_id"].map(node_key)
    parents["heading"] = pd.to_numeric(parents["heading"], errors="coerce")
    by_link = parents.set_index("link_key", drop=False)
    transitions = build_transition_counts(workers, rebuild_partitions)
    transition_lookup = transitions.set_index(["link_key", "next_link"])[
        "transition_count"
    ].to_dict()
    outgoing = parents.groupby("from_node_key")["link_key"].apply(list).to_dict()
    support = (
        transitions.groupby("link_key")["transition_count"].sum().sort_values(
            ascending=False
        )
        if not transitions.empty else pd.Series(dtype=float)
    )

    available = set(parents["link_key"])
    requested = [
        str(link) for link in paths.CORRIDOR_START_LINKS if str(link) in available
    ]
    corridor_rows = []
    used = set()
    considered = set()
    corridor_count = 0
    starts = chain(requested, support.index, parents["link_key"])
    for start in starts:
        if corridor_count >= paths.NUMBER_OF_CORRIDORS:
            break
        if start in used or start in considered:
            continue
        considered.add(start)
        path, supports, methods = trace_corridor(
            start, by_link, outgoing, transition_lookup, used
        )
        corridor_count += 1
        corridor_id = f"corridor_{corridor_count:02d}"
        for order, (link, count, method) in enumerate(
            zip(path, supports, methods), start=1
        ):
            corridor_rows.append({
                "corridor_id": corridor_id,
                "parent_order": order,
                "link_key": link,
                "transition_support": count,
                "selection_method": method,
            })
        used.update(path)

    corridors = pd.DataFrame(corridor_rows).merge(
        parents.drop(columns=["from_node_key", "to_node_key"]),
        on="link_key", how="left", validate="many_to_one",
    )
    segment_columns = [
        "moe_segment_id", "link_key", "segment_index", "segment_length_ft",
        "parent_length_ft", "is_parent_start", "is_parent_end",
        "posted_speed_limit_mph",
    ]
    segments = pd.read_parquet(paths.SEGMENT_FILE, columns=segment_columns)
    corridor_segments = corridors[
        ["corridor_id", "parent_order", "link_key"]
    ].merge(segments, on="link_key", how="left", validate="one_to_many")
    corridor_segments = corridor_segments.sort_values(
        ["corridor_id", "parent_order", "segment_index"]
    ).reset_index(drop=True)
    corridor_segments["corridor_segment_order"] = (
        corridor_segments.groupby("corridor_id").cumcount() + 1
    )
    corridor_segments["distance_end_ft"] = corridor_segments.groupby(
        "corridor_id"
    )["segment_length_ft"].cumsum()
    corridor_segments["distance_start_ft"] = (
        corridor_segments["distance_end_ft"]
        - corridor_segments["segment_length_ft"]
    )

    for _, frame in corridors.groupby("corridor_id", sort=False):
        connected = all(
            node_key(frame.iloc[index - 1]["to_node_id"])
            == node_key(frame.iloc[index]["from_node_id"])
            for index in range(1, len(frame))
        )
        if not connected:
            raise RuntimeError("Corridor topology check failed.")

    corridors.to_parquet(
        paths.CORRIDOR_FILE, index=False, compression=paths.PARQUET_COMPRESSION
    )
    corridor_segments.to_parquet(
        paths.CORRIDOR_SEGMENT_FILE,
        index=False,
        compression=paths.PARQUET_COMPRESSION,
    )
    print("\nCORRIDORS COMPLETE")
    print(corridors.groupby("corridor_id").agg(
        parent_links=("link_key", "size"),
        length_ft=("parent_length_ft", "sum"),
    ).round(1).to_string())
    print("Saved:", paths.CORRIDOR_SEGMENT_FILE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int)
    parser.add_argument("--rebuild-partitions", action="store_true")
    args = parser.parse_args()
    build_corridors(args.workers, args.rebuild_partitions)


if __name__ == "__main__":
    paths.check_inputs()
    main()
