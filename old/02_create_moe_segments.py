import math

import numpy as np
import pandas as pd
import shapely
from shapely.ops import substring

import paths
from progress import Progress


# Split one projected parent geometry into ordered pieces of at most 200 feet.
def split_parent(row):
    line = shapely.from_wkb(row.geometry_wkb)
    length_ft = float(row.parent_length_ft)
    piece_count = max(1, math.ceil(length_ft / paths.SEGMENT_LENGTH_FT - 1e-12))
    records = []

    for piece_index in range(piece_count):
        start_ft = piece_index * paths.SEGMENT_LENGTH_FT
        end_ft = min((piece_index + 1) * paths.SEGMENT_LENGTH_FT, length_ft)
        piece = substring(
            line,
            start_ft / paths.M_TO_FT,
            end_ft / paths.M_TO_FT,
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


# Create the complete segment inventory and verify parent-length conservation.
def create_segments():
    if not paths.PARENT_FILE.is_file():
        raise FileNotFoundError(f"Run stage 01 first: {paths.PARENT_FILE}")

    parents = pd.read_parquet(paths.PARENT_FILE)
    records = []
    progress = Progress("Segmentation", paths.REPORT_INTERVAL_SECONDS)

    for number, row in enumerate(parents.itertuples(index=False), start=1):
        records.extend(split_parent(row))
        progress.report(number, len(parents))

    segments = pd.DataFrame.from_records(records)
    if segments.empty or segments["moe_segment_id"].duplicated().any():
        raise RuntimeError("Segment identifier check failed.")
    if not segments["segment_length_ft"].gt(0).all():
        raise RuntimeError("Nonpositive segment length found.")
    if not segments["segment_length_ft"].le(
        paths.SEGMENT_LENGTH_FT + 1e-6
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

    segments.to_parquet(
        paths.SEGMENT_FILE,
        index=False,
        compression=paths.PARQUET_COMPRESSION,
    )
    progress.report(len(parents), len(parents), force=True)
    print("\nSEGMENTATION COMPLETE")
    print(f"Parent links: {len(parents):,}")
    print(f"MOE segments: {len(segments):,}")
    print(f"Short remainders: {(segments['segment_length_ft'] < 199.999).sum():,}")
    print("Saved:", paths.SEGMENT_FILE)


if __name__ == "__main__":
    paths.check_inputs()
    create_segments()
