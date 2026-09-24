import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

import paths
from parallel_utils import resolve_workers


FIGURES = [
    ("avg_speed_mph", "Average Speed", "Speed (mph)", None),
    ("avg_travel_time_s", "Average Segment Travel Time", "Travel Time (s)", None),
    ("slow_movement_count", "Slow Movement Count", "Slow Observations", None),
    ("slow_movement_pct", "Slow Movement Percentage", "Slow Movement (%)", 100),
    ("DSH", "Degree of Speed Harmonization", "DSH", None),
    ("speed_reduction_pct", "Speed Reduction Percentage", "Speed Reduction (%)", 100),
]

MOE_COLUMNS = ["moe_segment_id", "time_bin"] + [item[0] for item in FIGURES]


# Choose a stable display maximum while ignoring missing and negative values.
def display_max(values, fixed_max):
    if fixed_max is not None:
        return fixed_max
    finite = values[np.isfinite(values) & (values >= 0)]
    if not len(finite):
        return 1.0
    value = float(np.nanpercentile(finite, 95))
    return value if value > 0 else 1.0


# Plot one metric by five-minute interval and physical segment length.
def plot_metric(corridor_id, segments, moe, metric, title, label, fixed_max):
    ordered = segments.sort_values("corridor_segment_order")
    segment_ids = ordered["moe_segment_id"].tolist()
    time_bins = sorted(moe["time_bin"].dropna().unique())
    if not time_bins:
        raise RuntimeError(f"No MOE time bins found for {corridor_id}.")
    matrix = moe.pivot(
        index="moe_segment_id", columns="time_bin", values=metric
    ).reindex(index=segment_ids, columns=time_bins)
    values = matrix.to_numpy(float)
    masked = np.ma.masked_invalid(values)
    y_edges = np.concatenate(([0.0], ordered["distance_end_ft"].to_numpy(float)))
    x_edges = np.arange(len(time_bins) + 1) * paths.TIME_BIN_MINUTES
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad("lightgray")
    norm = Normalize(vmin=0, vmax=display_max(values, fixed_max), clip=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    mesh = ax.pcolormesh(
        x_edges, y_edges, masked, cmap=cmap, norm=norm,
        shading="flat", rasterized=True,
    )
    parent_ends = ordered.loc[ordered["is_parent_end"], "distance_end_ft"]
    for boundary in parent_ends.iloc[:-1]:
        ax.axhline(boundary, color="black", linewidth=0.7, alpha=0.8)

    centers = x_edges[:-1] + paths.TIME_BIN_MINUTES / 2
    ax.set_xticks(centers)
    ax.set_xticklabels(
        [pd.Timestamp(value).strftime("%H:%M") for value in time_bins],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Distance Along Corridor (ft)")
    ax.set_title(f"{corridor_id}: {title}")
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label(label)
    fig.tight_layout()
    output = paths.FIGURE_DIR / f"{corridor_id}_{metric}.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


# Read only one corridor's MOE rows and create all six figures.
def visualize_one_corridor(corridor_id, corridor_segments):
    segment_ids = corridor_segments["moe_segment_id"].astype(str).tolist()
    moe = pd.read_parquet(
        paths.MOE_FILE,
        columns=MOE_COLUMNS,
        filters=[("moe_segment_id", "in", segment_ids)],
    )
    moe["time_bin"] = pd.to_datetime(moe["time_bin"])
    for metric, title, label, fixed_max in FIGURES:
        plot_metric(
            corridor_id, corridor_segments, moe,
            metric, title, label, fixed_max,
        )
    return corridor_id, len(segment_ids)


# Generate different corridor figure sets concurrently.
def visualize_corridors(workers=None):
    segments = pd.read_parquet(paths.CORRIDOR_SEGMENT_FILE)
    groups = [
        (corridor_id, frame.copy())
        for corridor_id, frame in segments.groupby("corridor_id", sort=True)
    ]
    if not groups:
        raise RuntimeError("No corridors are available for visualization.")
    workers = min(resolve_workers(workers), len(groups))
    print(f"Creating corridor figures with {workers} workers")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(visualize_one_corridor, corridor_id, frame): corridor_id
            for corridor_id, frame in groups
        }
        for future in as_completed(futures):
            corridor_id, segment_count = future.result()
            print(
                f"Created 6 figures for {corridor_id} "
                f"({segment_count:,} segments)"
            )
    print("Saved:", paths.FIGURE_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    visualize_corridors(args.workers)


if __name__ == "__main__":
    paths.check_inputs()
    main()
