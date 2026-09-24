import argparse
import gc
import math
import os
import re
import threading
import time

import numpy as np
import pandas as pd
import psycopg2
import shapely
from psycopg2.extras import execute_values
from pyproj import Transformer
from shapely.ops import substring

from pathlib import Path


def _load_dotenv():
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def _env_number(name, default, cast):
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    number = cast(value)
    if number <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return number


_load_dotenv()

import moes
from moes.common import pack_link_rows, sum_partials, write_packed
from progress import RunClock, format_duration

SOURCE_CRS = "EPSG:4326"
PROJECTED_CRS = "EPSG:3083"
M_TO_FT = 1 / 0.3048
SEGMENT_LENGTH_FT = _env_number("SEGMENT_LENGTH_FT", 200.0, float)
TIME_BIN_MINUTES = _env_number("TIME_BIN_MINUTES", 15, int)
PARALLEL_WORKERS = _env_number(
    "PARALLEL_WORKERS", max(1, (os.cpu_count() or 1) - 2), int
)


def setting(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} in .env")
    return value


def quote_ident(name):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Unsafe SQL identifier: {name}")
    return '"' + name + '"'


def connect(database):
    return psycopg2.connect(
        host=setting("DB_HOST"),
        port=int(setting("DB_PORT")),
        user=setting("DB_USER"),
        password=setting("DB_PASSWORD"),
        dbname=database,
    )


# Read every distinct link represented in the connected-vehicle table.
def collect_observed_link_keys():
    table = quote_ident(setting("CV_TABLE"))
    connection = connect(setting("CV_DATABASE"))
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT DISTINCT link_key
                FROM {table}
                WHERE link_key IS NOT NULL
                  AND btrim(link_key) <> ''
                """
            )
            keys = [row[0] for row in cursor.fetchall()]
    finally:
        connection.close()

    return pd.DataFrame({"link_key": sorted(set(keys))})


# Load observed short links and use free_speed as the posted speed limit.
def prepare_parent_links(link_keys):
    table = quote_ident(setting("LINK_TABLE"))
    connection = connect(setting("STATIC_DATABASE"))
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "CREATE TEMP TABLE observed_keys (link_key text PRIMARY KEY)"
            )
            execute_values(
                cursor,
                "INSERT INTO observed_keys (link_key) VALUES %s",
                [(key,) for key in link_keys],
                page_size=10_000,
            )
            cursor.execute(
                f"""
                SELECT s.link_key, s.from_node_id, s.to_node_id, s.name,
                       s.facility_type, s.heading, s.free_speed,
                       s.geometry::text
                FROM {table} AS s
                JOIN observed_keys AS o ON s.link_key = o.link_key
                """
            )
            rows = cursor.fetchall()
    finally:
        connection.close()

    if not rows:
        raise RuntimeError("No observed link was found in the short-link table.")

    parents = pd.DataFrame(rows, columns=[
        "link_key", "from_node_id", "to_node_id", "name", "facility_type",
        "heading", "free_speed", "geometry_json",
    ])
    parents = parents.drop_duplicates("link_key", keep="first")
    parents["link_key"] = parents["link_key"].astype(str)
    found = set(parents["link_key"])
    missing = set(link_keys) - found
    if missing:
        print(f"Warning: {len(missing):,} observed link keys were absent from the short-link table.")

    transformer = Transformer.from_crs(
SOURCE_CRS, PROJECTED_CRS, always_xy=True
    )
    geometry = shapely.from_geojson(
        parents["geometry_json"].to_numpy(object),
        on_invalid="ignore",
    )
    geometry = shapely.line_merge(geometry)
    geometry = shapely.transform(geometry, transformer.transform, interleaved=False)
    parents["parent_length_ft"] = shapely.length(geometry) * M_TO_FT
    parents["geometry_wkb"] = shapely.to_wkb(geometry)
    parents = parents.drop(columns="geometry_json")

    posted_speed = pd.to_numeric(parents["free_speed"], errors="coerce")
    usable_speed = posted_speed.notna() & (posted_speed > 0)
    parents["posted_speed_limit_mph"] = posted_speed.where(usable_speed)
    parents["speed_match_status"] = np.where(
        usable_speed, "matched", "missing_posted_speed"
    )

    valid_length = np.isfinite(parents["parent_length_ft"]) & (
        parents["parent_length_ft"] > 0
    )
    invalid_length = int((~valid_length).sum())
    if invalid_length:
        print(f"Warning: {invalid_length:,} links have no usable geometry and were skipped.")
        parents = parents.loc[valid_length].copy()
    if parents.empty:
        raise RuntimeError("Parent geometry length check failed.")

    return parents


# Split one projected parent geometry into ordered pieces of at most 200 feet.
def split_parent(row):
    line = shapely.from_wkb(row.geometry_wkb)
    length_ft = float(row.parent_length_ft)
    piece_count = max(1, math.ceil(length_ft / SEGMENT_LENGTH_FT - 1e-12))
    records = []

    for piece_index in range(piece_count):
        start_ft = piece_index * SEGMENT_LENGTH_FT
        end_ft = min((piece_index + 1) * SEGMENT_LENGTH_FT, length_ft)
        piece = substring(
            line,
            start_ft / M_TO_FT,
            end_ft / M_TO_FT,
        )
        segment_number = piece_index + 1
        segment_id = f"{row.link_key}__{segment_number:04d}"
        from_node = (
            f"osm:{row.from_node_id}"
            if piece_index == 0
            else f"moe:{row.link_key}:{piece_index:04d}"
        )
        to_node = (
            f"osm:{row.to_node_id}"
            if segment_number == piece_count
            else f"moe:{row.link_key}:{segment_number:04d}"
        )
        records.append({
            "moe_segment_id": segment_id,
            "link_key": str(row.link_key),
            "segment_index": segment_number,
            "parent_segment_count": piece_count,
            "parent_from_node_id": str(row.from_node_id),
            "parent_to_node_id": str(row.to_node_id),
            "moe_from_node_id": from_node,
            "moe_to_node_id": to_node,
            "parent_length_ft": length_ft,
            "segment_length_ft": end_ft - start_ft,
            "parent_start_ft": start_ft,
            "parent_end_ft": end_ft,
            "is_parent_start": piece_index == 0,
            "is_parent_end": segment_number == piece_count,
            "facility_type": row.facility_type,
            "posted_speed_limit_mph": row.posted_speed_limit_mph,
            "speed_match_status": row.speed_match_status,
            "geometry_wkb": shapely.to_wkb(piece),
        })
    return records


# Cut parent links into 200-ft segments and save that inventory.
def create_segments(parents):
    records = []
    for row in parents.itertuples(index=False):
        records.extend(split_parent(row))

    segments = pd.DataFrame.from_records(records)
    if segments.empty or segments["moe_segment_id"].duplicated().any():
        raise RuntimeError("Segment identifier check failed.")
    if not segments["segment_length_ft"].gt(0).all():
        raise RuntimeError("Nonpositive segment length found.")
    if not segments["segment_length_ft"].le(
SEGMENT_LENGTH_FT + 1e-6
    ).all():
        raise RuntimeError("A segment exceeds the configured length.")

    coverage = segments.groupby("link_key", sort=False).agg(
        segment_total_ft=("segment_length_ft", "sum"),
        parent_length_ft=("parent_length_ft", "first"),
    )
    if not np.allclose(
        coverage["segment_total_ft"], coverage["parent_length_ft"], atol=1e-6
    ):
        raise RuntimeError("Segment lengths do not preserve parent lengths.")

    return segments.drop(columns="geometry_wkb")


def link_shards(link_keys, worker_count):
    hashed = pd.util.hash_pandas_object(
        pd.Series(link_keys, copy=False).astype(str),
        index=False,
    ).to_numpy()
    return (hashed % np.uint64(worker_count)).astype(np.int32)


def _geometry_lookup(parents):
    frame = parents[["link_key", "parent_length_ft", "geometry_wkb"]].reset_index(drop=True)
    frame["link_key"] = frame["link_key"].astype(str)
    geometry = shapely.from_wkb(frame.pop("geometry_wkb").to_numpy())
    frame["geometry"] = geometry
    index = pd.Series(frame.index.to_numpy(), index=frame["link_key"]).to_dict()
    return frame, index


def _page_count(table):
    connection = connect(setting("CV_DATABASE"))
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT GREATEST(pg_relation_size(%s::regclass) / current_setting('block_size')::bigint, 1)",
                (table.strip('"'),),
            )
            return int(cursor.fetchone()[0])
    finally:
        connection.close()


def _page_ranges(page_count, workers):
    workers = min(workers, page_count)
    edges = [round(i * page_count / workers) for i in range(workers + 1)]
    ranges = []
    for index in range(workers):
        start, end = edges[index], edges[index + 1]
        if index == workers - 1:
            ranges.append((start, None))
        elif end > start:
            ranges.append((start, end))
    return ranges


# Place each row on its link and choose the 200-ft piece.
def assign_points(data, parent_frame, parent_index):
    if data.empty or not parent_index:
        return data.iloc[0:0]
    data = data.copy()
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

    points = shapely.points(x / M_TO_FT, y / M_TO_FT)
    lines = parent_frame["geometry"].to_numpy()[indices]
    position_ft = shapely.line_locate_point(lines, points) * M_TO_FT
    length_ft = parent_frame["parent_length_ft"].to_numpy(float)[indices]
    position_ft = np.clip(position_ft, 0.0, np.nextafter(length_ft, -np.inf))
    segment_index = np.floor(position_ft / SEGMENT_LENGTH_FT).astype(int) + 1
    data["moe_segment_id"] = (
        data["link_key"] + "__"
        + pd.Series(segment_index, index=data.index).astype(str).str.zfill(4)
    )
    data["local_time"] = pd.to_datetime(data["local_time"], errors="coerce")
    data = data.dropna(subset=["local_time"])
    data["time_bin"] = data["local_time"].dt.floor(f"{TIME_BIN_MINUTES}min")
    return data


def _measure_names():
    return [module.TABLE_NAME for module in moes.POINT_MEASURES] + [
        moes.link_speed.TABLE_NAME
    ]


def _measures_from_attributes(attributes, segments):
    packed = {}
    for module in moes.POINT_MEASURES:
        values = module.from_attributes(attributes, segments)
        packed[module.TABLE_NAME] = pack_link_rows(segments, values)
    packed[moes.link_speed.TABLE_NAME] = moes.link_speed.pack(
        moes.link_speed.from_attributes(attributes, segments)
    )
    return packed


def _scan_page_range(table, start, end, parent_frame, parent_index):
    rows_read = 0
    totals = None
    if end is None:
        page_filter = "ctid >= %s::tid"
        parameters = (f"({start},0)",)
    else:
        page_filter = "ctid >= %s::tid AND ctid < %s::tid"
        parameters = (f"({start},0)", f"({end},0)")
    read_seconds = 0.0
    process_seconds = 0.0
    connection = connect(setting("CV_DATABASE"))
    try:
        with connection.cursor(name=f"points_{start}") as cursor:
            cursor.itersize = 100_000
            started = time.perf_counter()
            cursor.execute(
                f"""
                SELECT link_key, local_time, speed_mph, x_ft, y_ft
                FROM {table}
                WHERE {page_filter}
                  AND link_key IS NOT NULL
                  AND btrim(link_key) <> ''
                """,
                parameters,
            )
            read_seconds += time.perf_counter() - started
            while True:
                started = time.perf_counter()
                fetched = cursor.fetchmany(100_000)
                read_seconds += time.perf_counter() - started
                if not fetched:
                    break
                started = time.perf_counter()
                rows_read += len(fetched)
                frame = pd.DataFrame.from_records(
                    fetched,
                    columns=["link_key", "local_time", "speed_mph", "x_ft", "y_ft"],
                )
                totals = moes.attributes.add(
                    totals, assign_points(frame, parent_frame, parent_index)
                )
                process_seconds += time.perf_counter() - started
    finally:
        connection.close()
    return rows_read, totals, read_seconds, process_seconds


# Read disjoint slices of the trajectory table at the same time.
def assign_attributes(parents, segments):
    worker_count = min(PARALLEL_WORKERS, max(1, len(parents)))
    parents = parents.reset_index(drop=True)
    segments = segments.reset_index(drop=True)
    lookup_started = time.perf_counter()
    parent_frame, parent_index = _geometry_lookup(parents)
    process_seconds = time.perf_counter() - lookup_started
    del parents
    gc.collect()
    table = quote_ident(setting("CV_TABLE"))
    ranges = _page_ranges(_page_count(setting("CV_TABLE")), worker_count)
    errors = []
    results = []
    lock = threading.Lock()

    def run(start, end):
        try:
            result = _scan_page_range(table, start, end, parent_frame, parent_index)
        except Exception as exc:
            with lock:
                errors.append(exc)
            return
        with lock:
            results.append(result)

    threads = []
    for start, end in ranges:
        thread = threading.Thread(
            target=run, args=(start, end), name=f"segment-attributes-{start}", daemon=True
        )
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]

    rows_read = 0
    read_seconds = 0.0
    totals = None
    merge_started = time.perf_counter()
    for count, partial, read_part, process_part in results:
        rows_read += count
        read_seconds += read_part
        process_seconds += process_part
        totals = sum_partials(totals, partial, moes.attributes.COLUMNS)
        del partial
    process_seconds += time.perf_counter() - merge_started
    del parent_frame, parent_index, results
    gc.collect()
    worker_time = read_seconds + process_seconds
    read_share = 100 * read_seconds / worker_time if worker_time else 0
    print(
        f"Database read: {format_duration(read_seconds)} across {len(ranges)} workers "
        f"({read_share:.0f}% of worker time), {rows_read:,} rows"
    )

    calc_started = time.perf_counter()
    packed = {name: [] for name in _measure_names()}
    if totals is None or totals.empty:
        return packed, rows_read, len(ranges)
    link_key = totals["moe_segment_id"].str.rsplit("__", n=1).str[0]
    attribute_shards = link_shards(link_key, len(ranges))
    segment_shards = link_shards(segments["link_key"], len(ranges))
    for shard in range(len(ranges)):
        shard_attributes = totals.iloc[np.flatnonzero(attribute_shards == shard)]
        shard_segments = segments.iloc[np.flatnonzero(segment_shards == shard)]
        for name, frame in _measures_from_attributes(shard_attributes, shard_segments).items():
            if frame is not None and not frame.empty:
                packed[name].append(frame)
    process_seconds += time.perf_counter() - calc_started
    worker_time = read_seconds + process_seconds
    process_share = 100 * process_seconds / worker_time if worker_time else 0
    print(
        f"Processing: {format_duration(process_seconds)} across {len(ranges)} workers "
        f"({process_share:.0f}% of worker time)"
    )
    return packed, rows_read, len(ranges)


def main():
    parser = argparse.ArgumentParser(
        description="Segment the network and store five-minute link MOEs."
    )
    parser.parse_args()
    clock = RunClock()
    with clock.stage("Distinct link keys") as stage:
        observed = collect_observed_link_keys()
        stage.detail = f"{len(observed):,} links"
    with clock.stage("Short-link attributes") as stage:
        parents = prepare_parent_links(observed["link_key"].tolist())
        matched = int((parents["speed_match_status"] == "matched").sum())
        stage.detail = f"{len(parents):,} links, {matched:,} with posted speed"
    with clock.stage(f"{SEGMENT_LENGTH_FT:g}-ft segmentation") as stage:
        segments = create_segments(parents)
        stage.detail = f"{len(segments):,} segments"
    with clock.stage(f"Segment attributes on {PARALLEL_WORKERS} cores") as stage:
        packed, rows_read, worker_count = assign_attributes(parents, segments)
        stage.detail = f"{rows_read:,} rows on {worker_count} cores"
    write_seconds = 0.0
    with clock.stage("Database write") as stage:
        for module in moes.POINT_MEASURES:
            write_started = time.perf_counter()
            write_packed(module.TABLE_NAME, packed.pop(module.TABLE_NAME))
            elapsed = time.perf_counter() - write_started
            write_seconds += elapsed
            print(f"{module.TABLE_NAME}: {format_duration(elapsed)}")
        write_started = time.perf_counter()
        moes.link_speed.write_rows(packed.pop(moes.link_speed.TABLE_NAME))
        elapsed = time.perf_counter() - write_started
        write_seconds += elapsed
        print(f"{moes.link_speed.TABLE_NAME}: {format_duration(elapsed)}")
        stage.detail = f"{len(_measure_names())} tables in {format_duration(write_seconds)}"
    clock.summary()


if __name__ == "__main__":
    main()
