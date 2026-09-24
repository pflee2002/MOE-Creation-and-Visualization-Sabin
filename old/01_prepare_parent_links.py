import argparse

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from pyproj import Transformer

import paths
from progress import Progress


PARENT_COLUMNS = [
    "link_key", "from_node_id", "to_node_id", "name", "facility_type",
    "heading", "Matched_GID", "geometry",
]


# Read every distinct OSM link represented in the CV observations.
def collect_observed_link_keys(max_cv_rows=None):
    observed = set()
    rows_read = 0
    progress = Progress("CV link scan", paths.REPORT_INTERVAL_SECONDS)

    reader = pd.read_csv(
        paths.CV_FILE,
        usecols=["link_key"],
        dtype={"link_key": "string"},
        chunksize=paths.CSV_CHUNK_SIZE,
    )
    for chunk in reader:
        if max_cv_rows is not None:
            remaining = max_cv_rows - rows_read
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining]

        rows_read += len(chunk)
        keys = chunk["link_key"].dropna().astype(str).str.strip()
        observed.update(keys[keys.ne("")].unique())
        progress.report(rows_read)

    result = pd.DataFrame({"link_key": sorted(observed)})
    result.to_parquet(
        paths.OBSERVED_LINKS_FILE,
        index=False,
        compression=paths.PARQUET_COMPRESSION,
    )
    progress.report(rows_read, force=True)
    print(f"Observed links: {len(result):,}")
    return result


# Prepare projected TxDOT geometry and speed attributes for repeated use.
def prepare_txdot_reference(force=False):
    if paths.TXDOT_REFERENCE_FILE.is_file() and not force:
        rows = pq.read_metadata(paths.TXDOT_REFERENCE_FILE).num_rows
        print(f"Using existing TxDOT reference: {rows:,} rows")
        return

    transformer = Transformer.from_crs(
        paths.SOURCE_CRS, paths.PROJECTED_CRS, always_xy=True
    )
    schema = pa.schema([
        ("source_row", pa.int64()),
        ("txdot_link_id", pa.string()),
        ("txdot_gid", pa.float64()),
        ("heading_deg", pa.float64()),
        ("posted_speed_limit_mph", pa.float64()),
        ("geometry_wkb", pa.binary()),
        ("usable_for_speed", pa.bool_()),
    ], metadata={b"geometry_crs": paths.PROJECTED_CRS.encode()})

    temporary = paths.TXDOT_REFERENCE_FILE.with_suffix(".partial")
    progress = Progress("TxDOT reference", paths.REPORT_INTERVAL_SECONDS)
    total_rows = 0
    usable_rows = 0
    reader = pd.read_csv(
        paths.TXDOT_FILE,
        usecols=["LinkID", "GID", "Heading", "SpeedLimit", "Geometry"],
        dtype="string",
        chunksize=paths.CSV_CHUNK_SIZE,
    )

    with pq.ParquetWriter(
        temporary, schema, compression=paths.PARQUET_COMPRESSION
    ) as writer:
        for chunk in reader:
            count = len(chunk)
            gid = pd.to_numeric(chunk["GID"], errors="coerce").to_numpy(float)
            heading = pd.to_numeric(
                chunk["Heading"], errors="coerce"
            ).to_numpy(float)
            speed = pd.to_numeric(
                chunk["SpeedLimit"], errors="coerce"
            ).to_numpy(float)
            geometry = shapely.from_wkt(
                chunk["Geometry"].to_numpy(object, na_value=None),
                on_invalid="ignore",
            )
            geometry = shapely.transform(
                geometry, transformer.transform, interleaved=False
            )
            usable = (
                np.isfinite(gid) & (gid > 0) & (gid == np.floor(gid))
                & np.isfinite(speed) & (speed > 0)
                & ~shapely.is_missing(geometry) & ~shapely.is_empty(geometry)
            )
            table = pa.Table.from_pydict({
                "source_row": np.arange(total_rows, total_rows + count),
                "txdot_link_id": chunk["LinkID"].tolist(),
                "txdot_gid": gid,
                "heading_deg": heading,
                "posted_speed_limit_mph": speed,
                "geometry_wkb": shapely.to_wkb(geometry),
                "usable_for_speed": usable,
            }, schema=schema)
            writer.write_table(table)
            total_rows += count
            usable_rows += int(usable.sum())
            progress.report(total_rows)

    if pq.read_metadata(temporary).num_rows != total_rows:
        raise RuntimeError("TxDOT reference row-count check failed.")
    temporary.replace(paths.TXDOT_REFERENCE_FILE)
    progress.report(total_rows, force=True)
    print(f"Usable TxDOT speed candidates: {usable_rows:,}")


# Load usable TxDOT candidates once for batch matching.
def load_txdot_candidates():
    columns = [
        "source_row", "txdot_link_id", "txdot_gid", "heading_deg",
        "posted_speed_limit_mph", "geometry_wkb",
    ]
    candidates = pd.read_parquet(
        paths.TXDOT_REFERENCE_FILE,
        columns=columns,
        filters=[("usable_for_speed", "==", True)],
    )
    candidates["txdot_gid"] = candidates["txdot_gid"].astype("int64")
    candidates["txdot_geometry"] = shapely.from_wkb(
        candidates.pop("geometry_wkb").to_numpy()
    )
    return candidates


# Normalize positive integer TxDOT GIDs stored as text or decimal strings.
def normalize_gid(values):
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna() & (numeric > 0) & (numeric == np.floor(numeric))
    return numeric.where(valid).astype("Int64")


# Calculate the smallest circular difference between two headings.
def heading_difference(a, b):
    difference = np.abs(a - b)
    return np.minimum(difference, 360.0 - difference)


# Match one OSM batch to eligible TxDOT rows with the existing rules.
def match_speed_limits(parents, candidates):
    result = parents.copy()
    result["txdot_gid"] = normalize_gid(result["Matched_GID"])
    result["posted_speed_limit_mph"] = np.nan
    result["txdot_link_id"] = pd.NA
    result["txdot_match_distance_ft"] = np.nan
    result["txdot_heading_difference_deg"] = np.nan
    result["speed_match_status"] = np.where(
        result["txdot_gid"].isna(), "invalid_matched_gid", "no_valid_speed_for_gid"
    )

    valid = result["txdot_gid"].notna() & ~shapely.is_missing(
        result["geometry_projected"].to_numpy()
    )
    if not valid.any():
        return result

    left = result.loc[valid, ["source_order", "txdot_gid", "heading"]].copy()
    left["txdot_gid"] = left["txdot_gid"].astype("int64")
    gids = left["txdot_gid"].unique()
    right = candidates[candidates["txdot_gid"].isin(gids)]
    joined = left.merge(right, on="txdot_gid", how="left", sort=False)
    has_candidate = joined["source_row"].notna()
    joined = joined[has_candidate].copy()
    if joined.empty:
        return result

    parent_geometry = result.set_index("source_order")["geometry_projected"]
    joined["parent_geometry"] = joined["source_order"].map(parent_geometry)
    joined["heading_difference"] = heading_difference(
        pd.to_numeric(joined["heading"], errors="coerce").to_numpy(float),
        joined["heading_deg"].to_numpy(float),
    )
    compatible = (
        ~np.isfinite(joined["heading_difference"])
        | (joined["heading_difference"] <= paths.MAX_TXDOT_HEADING_DIFFERENCE_DEG)
    )
    compatible_orders = set(joined.loc[compatible, "source_order"])
    candidate_orders = set(joined["source_order"])
    no_heading = candidate_orders - compatible_orders
    result.loc[
        result["source_order"].isin(no_heading), "speed_match_status"
    ] = "no_heading_compatible_speed"

    joined = joined[compatible].copy()
    if joined.empty:
        return result

    midpoints = shapely.line_interpolate_point(
        joined["parent_geometry"].to_numpy(), 0.5, normalized=True
    )
    joined["distance_ft"] = (
        shapely.distance(midpoints, joined["txdot_geometry"].to_numpy())
        * paths.M_TO_FT
    )
    joined = joined.sort_values(
        ["source_order", "distance_ft", "heading_difference", "source_row"],
        na_position="last",
    ).drop_duplicates("source_order", keep="first")

    best = joined.set_index("source_order")
    lookup = result["source_order"]
    result["txdot_link_id"] = lookup.map(best["txdot_link_id"])
    result["txdot_match_distance_ft"] = lookup.map(best["distance_ft"])
    result["txdot_heading_difference_deg"] = lookup.map(
        best["heading_difference"]
    )
    within = result["txdot_match_distance_ft"] <= paths.MAX_TXDOT_MATCH_DISTANCE_FT
    result.loc[
        result["source_order"].isin(best.index) & ~within, "speed_match_status"
    ] = "txdot_segment_too_far"
    result.loc[within, "posted_speed_limit_mph"] = lookup[within].map(
        best["posted_speed_limit_mph"]
    )
    result.loc[within, "speed_match_status"] = "matched"
    return result


# Retrieve every observed parent link and attach its posted speed limit.
def prepare_parent_links(max_cv_rows=None):
    paths.check_inputs()
    observed = collect_observed_link_keys(max_cv_rows)
    observed_set = set(observed["link_key"])
    candidates = load_txdot_candidates()
    transformer = Transformer.from_crs(
        paths.SOURCE_CRS, paths.PROJECTED_CRS, always_xy=True
    )
    parts = []
    rows_scanned = 0
    source_order = 0
    progress = Progress("OSM scan", paths.REPORT_INTERVAL_SECONDS)

    reader = pd.read_csv(
        paths.OSM_FILE,
        usecols=PARENT_COLUMNS,
        dtype={"link_key": "string", "Matched_GID": "string"},
        chunksize=paths.CSV_CHUNK_SIZE,
    )
    for chunk in reader:
        rows_scanned += len(chunk)
        selected = chunk[chunk["link_key"].isin(observed_set)].copy()
        if not selected.empty:
            selected = selected.drop_duplicates("link_key", keep="first")
            selected["source_order"] = np.arange(
                source_order, source_order + len(selected), dtype=np.int64
            )
            source_order += len(selected)
            geometry = shapely.from_wkt(
                selected.pop("geometry").to_numpy(object, na_value=None),
                on_invalid="ignore",
            )
            selected["geometry_projected"] = shapely.transform(
                geometry, transformer.transform, interleaved=False
            )
            selected["parent_length_ft"] = (
                shapely.length(selected["geometry_projected"].to_numpy())
                * paths.M_TO_FT
            )
            parts.append(match_speed_limits(selected, candidates))
        progress.report(rows_scanned)

    if not parts:
        raise RuntimeError("No observed CV link was found in the OSM network.")

    result = pd.concat(parts, ignore_index=True)
    result = result.drop_duplicates("link_key", keep="first")
    result["geometry_wkb"] = shapely.to_wkb(result.pop("geometry_projected"))
    found = set(result["link_key"])
    missing = observed_set - found
    if missing:
        print(f"Warning: {len(missing):,} observed link keys were absent from OSM.")
    if result["link_key"].duplicated().any():
        raise RuntimeError("Duplicate parent link keys remain.")
    if not np.isfinite(result["parent_length_ft"]).all() or (
        result["parent_length_ft"] <= 0
    ).any():
        raise RuntimeError("Parent geometry length check failed.")

    keep = [
        "link_key", "from_node_id", "to_node_id", "name", "facility_type",
        "heading", "Matched_GID", "txdot_gid", "parent_length_ft",
        "posted_speed_limit_mph", "txdot_link_id",
        "txdot_match_distance_ft", "txdot_heading_difference_deg",
        "speed_match_status", "geometry_wkb",
    ]
    result[keep].to_parquet(
        paths.PARENT_FILE, index=False, compression=paths.PARQUET_COMPRESSION
    )
    progress.report(rows_scanned, force=True)
    print("\nPARENT LINKS COMPLETE")
    print(f"Observed link keys: {len(observed_set):,}")
    print(f"Parent links saved: {len(result):,}")
    print(result["speed_match_status"].value_counts(dropna=False).to_string())
    print("Saved:", paths.PARENT_FILE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-txdot", action="store_true")
    parser.add_argument("--max-cv-rows", type=int)
    args = parser.parse_args()
    paths.check_inputs()
    prepare_txdot_reference(force=args.force_txdot)
    prepare_parent_links(max_cv_rows=args.max_cv_rows)


if __name__ == "__main__":
    main()
