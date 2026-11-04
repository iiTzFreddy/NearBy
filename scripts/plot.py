#!/usr/bin/env python3
"""Turn the benchmark CSVs into the project's graphs.

Usage: python scripts/plot.py [--input results/benchmark.csv]
                              [--tuning results/tuning.csv] [--out results]
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Each structure keeps its colour and marker on every graph, so a graph that shows
# fewer structures doesn't repaint the rest. The marker keeps lines distinct
# without relying on colour.
STYLES = {
    "KdTree": {"color": "#2a78d6", "marker": "o"},
    "QuadTree": {"color": "#eb6834", "marker": "s"},
    "BruteForce": {"color": "#1baf7a", "marker": "^"},
}
FALLBACK_STYLE = {"color": "#52514e", "marker": "D"}
STRUCTURE_ORDER = list(STYLES)

INK = "#0b0b0b"
MUTED_INK = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
HEATMAP = LinearSegmentedColormap.from_list("nearby_blues", ["#e3eefb", "#2a78d6", "#0d3a73"])

DATASET_LABELS = {
    "gowalla": "Gowalla (real)",
    "uniform": "Uniform (synthetic)",
    "clustered": "Clustered (synthetic)",
}


def read_rows(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["n"] = int(row["n"])
        row["param"] = float(row["param"])
        for field in ("build_ms", "mean_us", "p95_us", "memory_bytes"):
            row[field] = float(row[field])
    return rows


def structures_in(rows):
    found = {row["structure"] for row in rows}
    known = [name for name in STRUCTURE_ORDER if name in found]
    return known + sorted(found - set(known))


def param_values(rows, query_type):
    return sorted({row["param"] for row in rows if row["query_type"] == query_type})


def main_dataset(rows):
    found = {row["dataset"] for row in rows}
    return "gowalla" if "gowalla" in found else sorted(found)[0]


def style_axes(ax, title, ylabel, xlabel="Number of points (n)"):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED_INK)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED_INK)
    ax.tick_params(colors=MUTED_INK, labelsize=8, which="both", length=0)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def draw_lines(ax, rows, value):
    """One line per structure. Values of zero can't appear on a log axis, so they're skipped.

    Returns where each line ends, for label_line_ends.
    """
    ends = []
    for structure in structures_in(rows):
        points = sorted(
            (row["n"], value(row))
            for row in rows
            if row["structure"] == structure and value(row) > 0
        )
        if not points:
            continue
        style = STYLES.get(structure, FALLBACK_STYLE)
        xs, ys = zip(*points)
        ax.plot(xs, ys, label=structure, color=style["color"], marker=style["marker"],
                markersize=5, linewidth=2, markeredgecolor=SURFACE, markeredgewidth=0.8)
        ends.append((structure, xs[-1], ys[-1]))
    ax.margins(x=0.12)
    return ends


def label_line_ends(ax, ends):
    """Name each line at its right end, pushing labels apart so close lines stay readable.

    Call this after the axis scales are set, because it measures on-screen positions.
    """
    min_gap_points = 11
    points_per_pixel = 72 / ax.figure.dpi
    placed = sorted(
        (ax.transData.transform((x, y))[1] * points_per_pixel, name, x, y)
        for name, x, y in ends
    )

    previous = None
    for height, name, x, y in placed:
        label_height = height if previous is None else max(height, previous + min_gap_points)
        ax.annotate(name, (x, y), xytext=(6, label_height - height), textcoords="offset points",
                    va="center", fontsize=8, color=INK, annotation_clip=False)
        previous = label_height


def add_legend(fig, axes):
    handles = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles[label] = handle
    if len(handles) >= 2:
        ordered = [name for name in STRUCTURE_ORDER if name in handles]
        ordered += sorted(set(handles) - set(ordered))
        fig.legend([handles[name] for name in ordered], ordered, loc="upper right",
                   ncol=len(ordered), frameon=False, fontsize=9, labelcolor=INK)


def save(fig, out_dir, name):
    path = out_dir / name
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def select(rows, dataset=None, query_type=None, param=None):
    return [
        row for row in rows
        if (dataset is None or row["dataset"] == dataset)
        and (query_type is None or row["query_type"] == query_type)
        and (param is None or row["param"] == param)
    ]


def one_row_per_build(rows):
    """Every row repeats its build time and memory, so keep one per (structure, dataset, n)."""
    unique = {(row["structure"], row["dataset"], row["n"]): row for row in rows}
    return list(unique.values())


def draw_build(ax, rows, dataset):
    ends = draw_lines(ax, one_row_per_build(select(rows, dataset)), lambda row: row["build_ms"])
    style_axes(ax, "Build time", "Build time (ms)")
    label_line_ends(ax, ends)


def draw_memory(ax, rows, dataset):
    ends = draw_lines(ax, one_row_per_build(select(rows, dataset)),
                      lambda row: row["memory_bytes"] / (1024 * 1024))
    style_axes(ax, "Memory", "Memory owned by the structure (MB)")
    label_line_ends(ax, ends)


def draw_query(ax, rows, dataset, query_type, param, title):
    ends = draw_lines(ax, select(rows, dataset, query_type, param), lambda row: row["mean_us"])
    style_axes(ax, title, "Mean query time (µs)")
    label_line_ends(ax, ends)


def closest(values, target):
    return min(values, key=lambda value: abs(value - target))


def plot_single(rows, out_dir, name, draw, dataset):
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    draw(ax, rows, dataset)
    add_legend(fig, [ax])
    save(fig, out_dir, name)


def plot_panels(out_dir, name, panels):
    """Small multiples on a shared y scale. Each panel is a function that draws on an axis."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), sharey=True,
                             squeeze=False, facecolor=SURFACE)
    axes = list(axes[0])
    for ax, panel in zip(axes, panels):
        panel(ax)
    for ax in axes[1:]:
        ax.set_ylabel("")
    add_legend(fig, axes)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out_dir, name)


def plot_knn(rows, out_dir, dataset):
    panels = [
        lambda ax, k=k: draw_query(ax, rows, dataset, "knn", k, f"k = {k:g}")
        for k in param_values(rows, "knn")
    ]
    plot_panels(out_dir, "knn_query_time.png", panels)


def plot_radius(rows, out_dir, dataset):
    panels = [
        lambda ax, r=r: draw_query(ax, rows, dataset, "radius", r, f"r = {r:g} km")
        for r in param_values(rows, "radius")
    ]
    plot_panels(out_dir, "radius_query_time.png", panels)


def plot_uniform_vs_clustered(rows, out_dir):
    k = closest(param_values(rows, "knn"), 10)
    found = {row["dataset"] for row in rows}
    panels = [
        lambda ax, d=dataset: draw_query(ax, rows, d, "knn", k, DATASET_LABELS[d])
        for dataset in ("uniform", "clustered", "gowalla") if dataset in found
    ]
    plot_panels(out_dir, "uniform_vs_clustered.png", panels)


def plot_report_summary(rows, out_dir, dataset):
    """The 2x2 grid for the report, which has only 5 pages to spend."""
    k = closest(param_values(rows, "knn"), 10)
    r = closest(param_values(rows, "radius"), 10)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), facecolor=SURFACE)
    draw_build(axes[0][0], rows, dataset)
    draw_memory(axes[0][1], rows, dataset)
    draw_query(axes[1][0], rows, dataset, "knn", k, f"kNN query time (k = {k:g})")
    draw_query(axes[1][1], rows, dataset, "radius", r, f"Radius query time (r = {r:g} km)")
    flat = [ax for row in axes for ax in row]
    add_legend(fig, flat)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out_dir, "report_summary.png")


def plot_tuning(path, out_dir):
    if not path.exists():
        print(f"skipped quadtree_tuning.png: {path} doesn't exist yet")
        return

    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    buckets = sorted({int(row["bucket"]) for row in rows})
    depths = sorted({int(row["depth"]) for row in rows})
    times = {(int(row["bucket"]), int(row["depth"])): float(row["knn_mean_us"]) for row in rows}
    grid = [[times.get((bucket, depth), float("nan")) for bucket in buckets] for depth in depths]

    fig, ax = plt.subplots(figsize=(6.5, 4), facecolor=SURFACE)
    image = ax.imshow(grid, cmap=HEATMAP, aspect="auto", origin="lower")
    ax.set_xticks(range(len(buckets)), labels=buckets)
    ax.set_yticks(range(len(depths)), labels=depths)
    ax.set_xlabel("Bucket size (B)", fontsize=9, color=MUTED_INK)
    ax.set_ylabel("Depth limit (D)", fontsize=9, color=MUTED_INK)
    ax.set_title("Quadtree tuning: mean kNN query time (µs), k = 10", loc="left", fontsize=11,
                 color=INK)
    ax.tick_params(colors=MUTED_INK, labelsize=8, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    low, high = image.get_clim()
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            if value != value:
                continue
            dark_cell = high > low and (value - low) / (high - low) > 0.45
            ax.text(x, y, f"{value:.3g}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if dark_cell else INK)
    save(fig, out_dir, "quadtree_tuning.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=Path("results/benchmark.csv"))
    parser.add_argument("--tuning", type=Path, default=Path("results/tuning.csv"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"{args.input} not found: run the benchmark first")

    rows = read_rows(args.input)
    if not rows:
        raise SystemExit(f"{args.input} has no results")
    args.out.mkdir(parents=True, exist_ok=True)
    dataset = main_dataset(rows)

    plot_single(rows, args.out, "build_time.png", draw_build, dataset)
    plot_knn(rows, args.out, dataset)
    plot_radius(rows, args.out, dataset)
    plot_single(rows, args.out, "memory.png", draw_memory, dataset)
    plot_tuning(args.tuning, args.out)
    plot_uniform_vs_clustered(rows, args.out)
    plot_report_summary(rows, args.out, dataset)


if __name__ == "__main__":
    main()
