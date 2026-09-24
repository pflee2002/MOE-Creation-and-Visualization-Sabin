import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import paths
from progress import Progress


CV_COLUMNS = [
    "journey_id", "capture_time", "local_time", "link_key",
    "speed_mph", "x_ft", "y_ft",
]

CV_SCHEMA = pa.schema([
    ("journey_id", pa.string()),
    ("capture_time", pa.float64()),
    ("local_time", pa.timestamp("ns")),
    ("link_key", pa.string()),
    ("speed_mph", pa.float64()),
    ("x_ft", pa.float64()),
    ("y_ft", pa.float64()),
])


# Resolve a safe process count without tying it to the number of partitions.
def resolve_workers(requested=None):
    workers = requested or paths.PARALLEL_WORKERS
    return max(1, min(int(workers), paths.JOURNEY_PARTITIONS))


# Remove only a named generated-work directory below the configured output root.
def reset_generated_directory(directory):
    directory = Path(directory).resolve()
    output_root = paths.OUTPUT_DIR.resolve()
    if output_root not in directory.parents or directory == output_root:
        raise RuntimeError(f"Refusing to reset unsafe directory: {directory}")
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)


# Describe the CV source so stale journey partitions are never reused silently.
def source_signature(max_cv_rows=None):
    stat = paths.CV_FILE.stat()
    return {
        "source": str(paths.CV_FILE.resolve()),
        "size": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
        "partition_count": paths.JOURNEY_PARTITIONS,
        "max_cv_rows": max_cv_rows,
    }


# Confirm every expected partition and its manifest still match the source file.
def partitions_are_current(directory, max_cv_rows=None):
    manifest_file = directory / "manifest.json"
    if not manifest_file.is_file():
        return False
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("signature") != source_signature(max_cv_rows):
        return False
    return all(
        (directory / f"journeys_{number:03d}.parquet").is_file()
        for number in range(paths.JOURNEY_PARTITIONS)
    )


# Normalize one CSV chunk before stable journey-based partitioning.
def normalize_cv_chunk(chunk):
    chunk = chunk[chunk["journey_id"].notna()].copy()
    if chunk.empty:
        return chunk
    chunk["journey_id"] = chunk["journey_id"].astype("string")
    chunk["link_key"] = chunk["link_key"].astype("string")
    for column in ["capture_time", "speed_mph", "x_ft", "y_ft"]:
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
    chunk["local_time"] = pd.to_datetime(chunk["local_time"], errors="coerce")
    return chunk


# Partition complete journeys once so later stages can process them independently.
def build_journey_partitions(rebuild=False, max_cv_rows=None):
    directory = (
        paths.JOURNEY_PARTITION_DIR
        if max_cv_rows is None
        else paths.OUTPUT_DIR / f"CV_Journey_Partitions_sample_{max_cv_rows}"
    )
    if not rebuild and partitions_are_current(directory, max_cv_rows):
        manifest = json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        print(
            f"Using existing journey partitions: "
            f"{manifest['rows_written']:,} rows in {paths.JOURNEY_PARTITIONS} files"
        )
        return sorted(directory.glob("journeys_*.parquet")), manifest["rows_written"]

    temporary = directory.with_name(directory.name + "_building")
    reset_generated_directory(temporary)
    writers = [None] * paths.JOURNEY_PARTITIONS
    files = [temporary / f"journeys_{number:03d}.parquet" for number in range(
        paths.JOURNEY_PARTITIONS
    )]
    rows_read = 0
    rows_written = 0
    progress = Progress("CV journey partitioning", paths.REPORT_INTERVAL_SECONDS)

    try:
        reader = pd.read_csv(
            paths.CV_FILE,
            usecols=CV_COLUMNS,
            dtype={"journey_id": "string", "link_key": "string"},
            chunksize=paths.CSV_CHUNK_SIZE,
        )
        for chunk in reader:
            if max_cv_rows is not None:
                remaining = max_cv_rows - rows_read
                if remaining <= 0:
                    break
                chunk = chunk.iloc[:remaining]
            rows_read += len(chunk)
            chunk = normalize_cv_chunk(chunk)
            if chunk.empty:
                progress.report(rows_read)
                continue

            hashes = pd.util.hash_pandas_object(
                chunk["journey_id"], index=False
            ).to_numpy(dtype=np.uint64)
            bucket = hashes % np.uint64(paths.JOURNEY_PARTITIONS)
            for number in np.unique(bucket):
                number = int(number)
                selected = chunk.loc[bucket == number, CV_COLUMNS]
                table = pa.Table.from_pandas(
                    selected, schema=CV_SCHEMA, preserve_index=False, safe=False
                )
                if writers[number] is None:
                    writers[number] = pq.ParquetWriter(
                        files[number], CV_SCHEMA,
                        compression=paths.PARQUET_COMPRESSION,
                    )
                writers[number].write_table(table)
                rows_written += len(selected)
            progress.report(rows_read)
    finally:
        for writer in writers:
            if writer is not None:
                writer.close()

    empty = pa.Table.from_pylist([], schema=CV_SCHEMA)
    for number, file in enumerate(files):
        if not file.exists():
            pq.write_table(empty, file, compression=paths.PARQUET_COMPRESSION)

    manifest = {
        "signature": source_signature(max_cv_rows),
        "rows_read": rows_read,
        "rows_written": rows_written,
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if directory.exists():
        shutil.rmtree(directory)
    temporary.replace(directory)
    progress.report(rows_read, force=True)
    print(
        f"Journey partitions complete: {rows_written:,} rows in "
        f"{paths.JOURNEY_PARTITIONS} files"
    )
    return sorted(directory.glob("journeys_*.parquet")), rows_written
