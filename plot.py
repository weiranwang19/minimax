"""Plot W&B run histories for publication figures.

Edit the globals below, then run:

    python plot.py
"""

from __future__ import annotations

import math
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "minimax-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "minimax-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import wandb


# W&B controls.
PROJECT = "minimax"
ENTITY = None  # Set to your W&B entity/workspace if it is not configured locally.

# Plot controls.
RUN_NAMES = [
    "fop-n400-m400-l20-inst9-seed0",
    "smo-n400-m400-l20-idx8-seed0",
    "gcmo-n400-m400-l20-idx8-seed0",
    "gcmo-n400-m400-l20-idx8-seed0-lip6",
]
DISPLAY_NAMES = [
    "FOP",
    "SMO",
    "Ours",
    r"Ours (Tuned $L_{\nabla h}$)",
]
# upper_obj, lower_gap, feas
METRIC = "ncwc/upper_obj" 
X_LIM = (0.0, 6.0)  # In millions of iterations. Use None for full range.

# Options: None, "log".
# "log" plots log10(max(abs(y), Y_TRANSFORM_FLOOR)), so lower is better.
Y_TRANSFORM = None
Y_TRANSFORM_FLOOR = 1e-14
SHOW_TRANSFORMED_Y_LABEL = False

# Figure sizing. These defaults are tuned for compact, square publication plots.
FIG_SIZE = 3.15
FONT_SIZE = 9
TICK_SIZE = 8
LEGEND_TEXT_SIZE = 7.5
LEGEND_BOX_SIZE = 0.28
LEGEND_HANDLE_LENGTH = 1.45
LEGEND_LOC = "best"
CURVE_SIZE = 2.0
MAX_POINTS_PER_CURVE = 2500

OUTPUT_DIR = Path("results")
OUTPUT_BASENAME = None

# Override this when plotting a metric that should use a different x column.
X_METRIC = None


def _project_path(api):
    if "/" in PROJECT:
        return PROJECT
    if ENTITY:
        return f"{ENTITY}/{PROJECT}"

    default_entity = getattr(api, "default_entity", None)
    if default_entity:
        return f"{default_entity}/{PROJECT}"

    raise RuntimeError(
        "Could not infer the W&B entity. Set ENTITY at the top of this file "
        'or set PROJECT to "entity/project".'
    )


def _normalize_display_names():
    if len(DISPLAY_NAMES) == len(RUN_NAMES):
        return list(DISPLAY_NAMES)

    if len(DISPLAY_NAMES) == 1 and "," in DISPLAY_NAMES[0]:
        names = [name.strip() for name in DISPLAY_NAMES[0].split(",")]
        if len(names) == len(RUN_NAMES):
            return names

    raise ValueError("DISPLAY_NAMES must have the same length as RUN_NAMES.")


def _x_metric_candidates():
    if X_METRIC:
        return [X_METRIC]

    candidates = []
    if METRIC.startswith("ncwc/"):
        candidates.extend(
            [
                "ncwc/cumulative_scsc_inner_iters",
                "ncwc/cumulative_inner_iters",
            ]
        )
    elif METRIC.startswith("fop/stage/"):
        candidates.append("fop/stage/index")
    elif METRIC.startswith("smo/stage/"):
        candidates.append("smo/stage/index")
    elif METRIC.startswith("gcmo/stage/"):
        candidates.append("gcmo/stage/index")

    candidates.append("_step")
    return candidates


def _find_run(api, project_path, run_name):
    runs = api.runs(
        project_path,
        filters={"displayName": run_name},
        order="-created_at",
        per_page=1,
    )
    match = next(iter(runs), None)
    if match is not None:
        return match

    runs = api.runs(
        project_path,
        filters={"name": run_name},
        order="-created_at",
        per_page=1,
    )
    match = next(iter(runs), None)
    if match is not None:
        return match

    raise ValueError(f'No W&B run named "{run_name}" found in {project_path}.')


def _coerce_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _history_xy(run):
    for x_key in _x_metric_candidates():
        rows_iter = iter(run.scan_history(keys=[METRIC, x_key], page_size=1000))
        rows = []
        for row in rows_iter:
            x_val = _coerce_float(row.get(x_key))
            y_val = _coerce_float(row.get(METRIC))
            if x_val is not None and y_val is not None:
                rows.append((x_val / 1_000_000.0, y_val))

        if rows:
            rows.sort(key=lambda item: item[0])
            return rows, x_key

    raise ValueError(f'Run "{run.name}" has no finite rows for metric "{METRIC}".')


def _clip_x_range(rows):
    if X_LIM is None:
        return rows

    xmin, xmax = X_LIM
    if xmin is None and xmax is None:
        return rows

    clipped = []
    for x_val, y_val in rows:
        if xmin is not None and x_val < xmin:
            continue
        if xmax is not None and x_val > xmax:
            continue
        clipped.append((x_val, y_val))
    return clipped


def _transform_y_value(y_val):
    if Y_TRANSFORM is None:
        return y_val

    if Y_TRANSFORM == "log":
        return math.log10(max(abs(y_val), Y_TRANSFORM_FLOOR))

    raise ValueError(f'Unknown Y_TRANSFORM "{Y_TRANSFORM}".')


def _transform_rows(rows):
    transformed = []
    for x_val, y_val in rows:
        transformed_y = _coerce_float(_transform_y_value(y_val))
        if transformed_y is not None:
            transformed.append((x_val, transformed_y))
    return transformed


def _downsample(rows):
    if MAX_POINTS_PER_CURVE is None or len(rows) <= MAX_POINTS_PER_CURVE:
        return rows

    stride = math.ceil(len(rows) / MAX_POINTS_PER_CURVE)
    sampled = rows[::stride]
    if sampled[-1] != rows[-1]:
        sampled.append(rows[-1])
    return sampled


def _sanitize_filename(value):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return safe or "metric"


def _output_path():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_BASENAME or f"plot_{_sanitize_filename(METRIC)}"
    if OUTPUT_BASENAME is None and Y_TRANSFORM is not None:
        stem = f"{stem}_{_sanitize_filename(Y_TRANSFORM)}"
    return OUTPUT_DIR / f"{stem}.pdf"


def main():
    display_names = _normalize_display_names()
    api = wandb.Api()
    project_path = _project_path(api)

    series = []
    for run_name, display_name in zip(RUN_NAMES, display_names):
        run = _find_run(api, project_path, run_name)
        rows, x_key = _history_xy(run)
        raw_last_x = rows[-1][0]
        rows = _clip_x_range(rows)
        if not rows:
            raise ValueError(
                f'Run "{run.name}" has no points inside X_LIM={X_LIM}. '
                "Set X_LIM=None or widen the range."
            )
        rows = _transform_rows(rows)
        rows = _downsample(rows)
        x_vals, y_vals = zip(*rows)
        series.append((display_name, x_vals, y_vals))
        print(
            f'{display_name}: {len(rows)} plotted points from "{run.name}" '
            f"using {x_key}; visible x={x_vals[-1]:.3g}M, raw last x={raw_last_x:.3g}M."
        )

    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_TEXT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE), constrained_layout=True)
    for display_name, x_vals, y_vals in series:
        ax.plot(x_vals, y_vals, linewidth=CURVE_SIZE, label=display_name)

    ax.set_xlabel("Million Iters")
    if SHOW_TRANSFORMED_Y_LABEL and Y_TRANSFORM == "log":
        ax.set_ylabel(r"$-\log_{10}(|\mathrm{metric}|)$")
    else:
        ax.set_ylabel("")
    if X_LIM is not None:
        ax.set_xlim(*X_LIM)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    ax.margins(x=0.02, y=0.06)
    ax.grid(True, linewidth=0.45, alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=3, width=0.8)
    ax.legend(
        loc=LEGEND_LOC,
        frameon=True,
        framealpha=0.92,
        borderpad=LEGEND_BOX_SIZE,
        handlelength=LEGEND_HANDLE_LENGTH,
        handletextpad=0.45,
        labelspacing=0.25,
    )

    output_path = _output_path()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
