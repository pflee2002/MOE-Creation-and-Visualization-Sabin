import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely

import paths


REQUIRED_COLUMNS = {
    "CV": (
        paths.CV_FILE,
        {"journey_id", "capture_time", "local_time", "link_key", "speed_mph", "x_ft", "y_ft"},
    ),
    "OSM": (
        paths.OSM_FILE,
        {"link_key", "from_node_id", "to_node_id", "geometry", "heading", "Matched_GID"},
    ),
    "TxDOT": (
        paths.TXDOT_FILE,
        {"LinkID", "GID", "Geometry", "Heading", "SpeedLimit"},
    ),
}


# Read only CSV headers and confirm every required field is present.
def check_headers():
    for label, (file, required) in REQUIRED_COLUMNS.items():
        with file.open("r", encoding="utf-8-sig", newline="") as stream:
            columns = set(next(csv.reader(stream)))
        missing = required - columns
        if missing:
            raise RuntimeError(f"{label} is missing columns: {sorted(missing)}")
        print(f"{label} columns: passed")


# Confirm an existing TxDOT reference can be resumed safely.
def check_txdot_reference():
    if not paths.TXDOT_REFERENCE_FILE.is_file():
        print("TxDOT reference: will be created during stage 01")
        return
    metadata = pq.read_metadata(paths.TXDOT_REFERENCE_FILE)
    required = {
        "txdot_link_id", "txdot_gid", "heading_deg",
        "posted_speed_limit_mph", "geometry_wkb", "usable_for_speed",
    }
    missing = required - set(metadata.schema.names)
    if metadata.num_rows <= 0 or missing:
        raise RuntimeError("Existing TxDOT reference is incomplete.")
    print(f"TxDOT reference: passed ({metadata.num_rows:,} rows)")


# Verify the defining 831-ft example produces four full pieces and one remainder.
def check_segmentation_example():
    script = Path(__file__).with_name("02_create_moe_segments.py")
    spec = importlib.util.spec_from_file_location("create_segments", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    length_ft = 831.0
    line = shapely.LineString([(0, 0), (length_ft / paths.M_TO_FT, 0)])
    row = SimpleNamespace(
        link_key="example", from_node_id="A", to_node_id="B",
        parent_length_ft=length_ft, geometry_wkb=shapely.to_wkb(line),
        facility_type="arterial", posted_speed_limit_mph=45.0,
        speed_match_status="matched",
    )
    pieces = module.split_parent(row)
    lengths = np.array([piece["segment_length_ft"] for piece in pieces])
    if len(pieces) != 5 or not np.allclose(lengths, [200, 200, 200, 200, 31]):
        raise RuntimeError("Synthetic 200-ft segmentation check failed.")
    if any(
        pieces[index]["moe_to_node_id"] != pieces[index + 1]["moe_from_node_id"]
        for index in range(len(pieces) - 1)
    ):
        raise RuntimeError("Synthetic segment connectivity check failed.")
    print("831-ft segmentation: passed (200, 200, 200, 200, 31)")


# Confirm nullable strings can be partitioned without changing journey aggregates.
def check_parallel_journey_logic():
    script = Path(__file__).with_name("03_calculate_segment_moes.py")
    spec = importlib.util.spec_from_file_location("calculate_segment_moes", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = pd.Timestamp("2026-01-01 19:00:00")
    rows = []
    for journey, offset in (("J1", 0), ("J2", 60)):
        for number, speed in enumerate((20.0, 25.0, 30.0)):
            rows.append({
                "journey_id": journey,
                "capture_time": offset + number * 10,
                "local_time": base + pd.Timedelta(seconds=offset + number * 10),
                "link_key": "A",
                "moe_segment_id": "A__0001",
                "parent_position_ft": number * 20.0,
                "parent_length_ft": 100.0,
                "speed_mph": speed,
            })
    data = pd.DataFrame(rows)
    data["journey_id"] = data["journey_id"].astype("string[pyarrow]")
    data["link_key"] = data["link_key"].astype("string[pyarrow]")
    data["moe_segment_id"] = data["moe_segment_id"].astype("string[pyarrow]")
    whole, _ = module.calculate_speed_components(module.prepare_sequence(data))
    parts = []
    for _, frame in data.groupby("journey_id"):
        part, _ = module.calculate_speed_components(module.prepare_sequence(frame))
        parts.append(part)
    divided = pd.concat(parts).groupby(
        ["moe_segment_id", "time_bin"], as_index=False
    ).agg(
        waypoint_count=("waypoint_count", "sum"),
        journey_count=("journey_count", "sum"),
        total_speed_sum=("total_speed_sum", "sum"),
    )
    expected = whole[[
        "moe_segment_id", "time_bin", "waypoint_count",
        "journey_count", "total_speed_sum",
    ]]
    pd.testing.assert_frame_equal(expected, divided, check_dtype=False)
    print("Journey-partition equivalence: passed")


def main():
    paths.check_inputs()
    check_headers()
    check_txdot_reference()
    check_segmentation_example()
    check_parallel_journey_logic()
    print("\nPIPELINE CHECKS PASSED")


if __name__ == "__main__":
    main()
