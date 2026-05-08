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
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, MaxNLocator, NullFormatter

import wandb


# W&B controls.
PROJECT = "minimax"
ENTITY = None  # Set to your W&B entity/workspace if it is not configured locally.

# Plot controls.9,8,8,8/1,0,0,0 (used)
RUN_NAMES = [
    "fop-n400-m400-l20-inst1-seed0",
    "smo-n400-m400-l20-idx0-seed0",
    "gcmo-n400-m400-l20-idx0-seed0",
    "gcmo-n400-m400-l20-idx0-seed0-lip6",
]
DISPLAY_NAMES = [
    "FOP",
    "SMO",
    "Ours",
    r"Ours (Tuned $L_{\nabla h}$)",
]
# Set METRIC to a W&B key string, e.g. "ncwc/lower_gap", or to a composite
# spec. Composite "value" functions receive a dict of finite float values.
# MAX_FEAS_LOWER_GAP_METRIC = {
#     "name": "ncwc/max_feas_lower_gap",
#     "keys": ["ncwc/feas", "ncwc/lower_gap"],
#     "value": lambda values: max(values["ncwc/feas"], values["ncwc/lower_gap"]),
# }
# METRIC = MAX_FEAS_LOWER_GAP_METRIC
METRIC = "ncwc/feas"
X_LIM = (0.0, 6.0)  # In millions of iterations. Use None for full range.
Y_LIM = None
# Y_LIM = (-0.01, 0.8)  # Use None for auto range, or e.g. (-8.0, 1.0).

# Options: None, "log".
# "log" uses a log-scaled y-axis via semilogy after flooring abs(y).
Y_TRANSFORM = "log"
# Y_TRANSFORM = None
Y_TRANSFORM_FLOOR = 1e-14
SHOW_TRANSFORMED_Y_LABEL = False

# Figure sizing. These defaults are tuned for compact, square publication plots.
FIG_SIZE_X = 3
FIG_SIZE_Y = 2
FONT_SIZE = 9
TICK_SIZE = 8
LEGEND_TEXT_SIZE = 9.0
LEGEND_FONT_WEIGHT = "bold"
LEGEND_BOX_SIZE = 0.8
LEGEND_HANDLE_LENGTH = 1.45
LEGEND_LOC = "best"
SHOW_LEGEND = False
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


def _metric_name():
    if isinstance(METRIC, str):
        return METRIC
    return METRIC["name"]


def _metric_keys():
    if isinstance(METRIC, str):
        return [METRIC]
    return list(METRIC["keys"])


def _metric_value(row):
    if isinstance(METRIC, str):
        return row.get(METRIC)

    values = {}
    for key in _metric_keys():
        value = _coerce_float(row.get(key))
        if value is None:
            return None
        values[key] = value
    return METRIC["value"](values)


def _x_metric_candidates():
    if X_METRIC:
        return [X_METRIC]

    metric_name = _metric_name()
    candidates = []
    if metric_name.startswith("ncwc/"):
        candidates.extend(
            [
                "ncwc/cumulative_scsc_inner_iters",
                "ncwc/cumulative_inner_iters",
            ]
        )
    elif metric_name.startswith("fop/stage/"):
        candidates.append("fop/stage/index")
    elif metric_name.startswith("smo/stage/"):
        candidates.append("smo/stage/index")
    elif metric_name.startswith("gcmo/stage/"):
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
        scan_keys = list(dict.fromkeys(_metric_keys() + [x_key]))
        rows_iter = iter(run.scan_history(keys=scan_keys, page_size=1000))
        rows = []
        for row in rows_iter:
            x_val = _coerce_float(row.get(x_key))
            y_val = _coerce_float(_metric_value(row))
            if x_val is not None and y_val is not None:
                rows.append((x_val / 1_000_000.0, y_val))

        if rows:
            rows.sort(key=lambda item: item[0])
            return rows, x_key

    raise ValueError(f'Run "{run.name}" has no finite rows for metric "{_metric_name()}".')


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
        return max(abs(y_val), Y_TRANSFORM_FLOOR)

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
    stem = OUTPUT_BASENAME or f"plot_{_sanitize_filename(_metric_name())}"
    if OUTPUT_BASENAME is None and Y_TRANSFORM is not None:
        stem = f"{stem}_{_sanitize_filename(Y_TRANSFORM)}"
    return OUTPUT_DIR / f"{stem}.pdf"


def _format_log10_exponent(value, _position):
    if value <= 0.0 or not math.isfinite(value):
        return ""

    exponent = math.log10(value)
    rounded_exponent = round(exponent)
    if not math.isclose(exponent, rounded_exponent, rel_tol=0.0, abs_tol=1e-10):
        return ""
    return f"{rounded_exponent:d}"


def _visible_even_log10_ticks(ax):
    ymin, ymax = ax.get_ylim()
    if ymin <= 0.0 or ymax <= 0.0:
        return []

    low_exponent = math.ceil(math.log10(ymin) - 1e-12)
    high_exponent = math.floor(math.log10(ymax) + 1e-12)
    first_even_exponent = 2 * math.ceil(low_exponent / 2)
    last_even_exponent = 2 * math.floor(high_exponent / 2)
    if first_even_exponent > last_even_exponent:
        return []
    return [10.0**exponent for exponent in range(first_even_exponent, last_even_exponent + 1, 2)]


def _configure_axis_ticks(ax):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    if Y_TRANSFORM == "log":
        ax.yaxis.set_major_locator(FixedLocator(_visible_even_log10_ticks(ax)))
        ax.yaxis.set_major_formatter(FuncFormatter(_format_log10_exponent))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10), numticks=60))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))


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

    fig, ax = plt.subplots(figsize=(FIG_SIZE_X, FIG_SIZE_Y), constrained_layout=True)
    plot_func = ax.semilogy if Y_TRANSFORM == "log" else ax.plot
    for display_name, x_vals, y_vals in series:
        plot_func(x_vals, y_vals, linewidth=CURVE_SIZE, label=display_name)

    ax.set_xlabel("Million Iters")
    if SHOW_TRANSFORMED_Y_LABEL and Y_TRANSFORM == "log":
        ax.set_ylabel(_metric_name())
    else:
        ax.set_ylabel("")
    if X_LIM is not None:
        ax.set_xlim(*X_LIM)
    if Y_LIM is not None:
        ax.set_ylim(*Y_LIM)
    _configure_axis_ticks(ax)
    ax.margins(x=0.02, y=0.06)
    ax.grid(True, linewidth=0.45, alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=3, width=0.8)
    if SHOW_LEGEND:
        ax.legend(
            loc=LEGEND_LOC,
            prop={"size": LEGEND_TEXT_SIZE, "weight": LEGEND_FONT_WEIGHT},
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
