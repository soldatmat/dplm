from __future__ import annotations

import math
import re
import textwrap
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

_VALIDATION_INFO_PATTERN = re.compile(r"\[byprot\.tasks\]\[INFO\].*Validation Info @")
_EPOCH_STEP_PATTERN = re.compile(r"Epoch (\d+), global step (\d+)\):")
_METRIC_PATTERN = re.compile(
    r"val/([A-Za-z0-9_]+)=([+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)"
)


def has_values(values: Iterable[float]) -> bool:
    return any(value == value for value in values)


def read_validation_log(log_file: str | Path) -> str:
    log_path = Path(log_file).expanduser()
    raw = log_path.read_text(encoding="utf-8")
    relevant_lines = [
        line for line in raw.splitlines() if _VALIDATION_INFO_PATTERN.search(line)
    ]
    return "\n".join(relevant_lines)


def extract_metrics(log: str) -> dict[str, object]:
    entries: list[tuple[int, int, dict[str, float]]] = []

    for line in log.strip().split("\n"):
        epoch_step_match = _EPOCH_STEP_PATTERN.search(line)
        if not epoch_step_match:
            continue

        epoch = int(epoch_step_match.group(1))
        step = int(epoch_step_match.group(2))
        metrics = {
            name: float(value) for name, value in _METRIC_PATTERN.findall(line)
        }
        entries.append((epoch, step, metrics))

    epochs = [epoch for epoch, _, _ in entries]
    global_steps = [step for _, step, _ in entries]

    available_metrics = sorted(
        {name for _, _, metrics in entries for name in metrics}
    )

    metric_series: dict[str, list[float]] = {}
    for metric_name in available_metrics:
        metric_series[metric_name] = [
            metrics.get(metric_name, float("nan")) for _, _, metrics in entries
        ]

    return {
        "epochs": epochs,
        "global_steps": global_steps,
        "series": metric_series,
        "available_metrics": available_metrics,
    }


def normalize_runs(runs: list[str | dict[str, str]]) -> list[dict[str, Path | str]]:
    if not runs:
        raise ValueError("Define at least one run path in `runs`.")

    normalized_runs: list[dict[str, Path | str]] = []
    for run in runs:
        if isinstance(run, dict):
            if "path" not in run:
                raise KeyError("Each run dict must define a `path`.")
            run_path = Path(run["path"]).expanduser()
            label = run.get("label", run_path.name)
        else:
            run_path = Path(run).expanduser()
            label = run_path.name

        normalized_runs.append({"label": label, "path": run_path})

    return normalized_runs


def load_run_metric(
    run_path: str | Path,
    metric_name: str,
    x_axis: str = "global_steps",
) -> tuple[list[int], list[float]]:
    run_path = Path(run_path).expanduser()
    log_file = run_path / "train.log"
    if not log_file.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")

    parsed = extract_metrics(read_validation_log(log_file))
    x_values = parsed[x_axis]
    series = parsed["series"]

    if metric_name not in series:
        raise KeyError(
            f"Metric 'val/{metric_name}' not found in {log_file}. "
            f"Available: {parsed['available_metrics']}"
        )

    return x_values, series[metric_name]


def _compute_shared_ylim(series_per_run: list[list[float]]) -> tuple[float, float]:
    finite_values = [
        value
        for values in series_per_run
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(value)
    ]
    if not finite_values:
        return (0.0, 1.0)

    y_min = min(finite_values)
    y_max = max(finite_values)

    if y_min == y_max:
        padding = 0.05 * (abs(y_min) if y_min != 0 else 1.0)
        return (y_min - padding, y_max + padding)

    padding = 0.05 * (y_max - y_min)
    return (y_min - padding, y_max + padding)


def _format_subplot_title(
    title: str,
    max_total_chars: int,
    wrap_width: int = 26,
    max_lines: int = 2,
) -> str:
    normalized = title.replace("_", " ").replace("-", " ")
    clipped = (
        normalized
        if len(normalized) <= max_total_chars
        else normalized[: max_total_chars - 3] + "..."
    )
    wrapped_lines = textwrap.wrap(
        clipped,
        width=wrap_width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not wrapped_lines:
        wrapped_lines = [clipped]
    if len(wrapped_lines) > max_lines:
        wrapped_lines = wrapped_lines[:max_lines]
        wrapped_lines[-1] = wrapped_lines[-1][: max(0, wrap_width - 3)] + "..."
    return "\n".join(wrapped_lines)


def _choose_tick_scale(max_abs_value: float) -> tuple[float, str]:
    if max_abs_value >= 1_000_000:
        return 1_000_000.0, "M"
    if max_abs_value >= 1_000:
        return 1_000.0, "k"
    return 1.0, ""


def _make_compact_tick_formatter(max_abs_value: float) -> FuncFormatter:
    scale, suffix = _choose_tick_scale(max_abs_value)

    def _fmt(value: float, _pos: float) -> str:
        if abs(value) < 1e-12:
            return "0"

        scaled = value / scale
        if abs(scaled) >= 100:
            text = f"{scaled:.0f}"
        elif abs(scaled) >= 10:
            text = f"{scaled:.1f}"
            text = text.rstrip("0").rstrip(".")
        else:
            text = f"{scaled:.2f}"
            text = text.rstrip("0").rstrip(".")
        return f"{text}{suffix}"

    return FuncFormatter(_fmt)


def plot_metric_comparison(
    runs: list[str | dict[str, str]],
    metric_name: str,
    x_axis: str = "global_steps",
    ncols: int = 3,
    height_per_row: float = 4.6,
    width_per_col: float = 6.4,
    max_title_length: int = 60,
    shared_y_axis: bool = True,
    title_wrap_width: int = 26,
    title_max_lines: int = 2,
    wspace: float = 0.28,
    hspace: float = 0.62,
) -> plt.Figure:
    normalized_runs = normalize_runs(runs)

    x_per_run: list[list[int]] = []
    y_per_run: list[list[float]] = []

    for run in normalized_runs:
        x_values, y_values = load_run_metric(run["path"], metric_name, x_axis=x_axis)
        x_per_run.append(x_values)
        y_per_run.append(y_values)

    n_runs = len(normalized_runs)
    nrows = (n_runs + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * width_per_col, nrows * height_per_row),
        squeeze=False,
        constrained_layout=False,
    )
    flat_axes = axes.flatten()

    y_limits = _compute_shared_ylim(y_per_run) if shared_y_axis else None
    all_x_values = [x for series in x_per_run for x in series]
    max_abs_x = max((abs(float(x)) for x in all_x_values), default=0.0)
    x_formatter = _make_compact_tick_formatter(max_abs_x)

    for index, run in enumerate(normalized_runs):
        ax = flat_axes[index]
        label = str(run["label"])
        x_values = x_per_run[index]
        y_values = y_per_run[index]

        ax.plot(x_values, y_values, linewidth=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Global steps" if x_axis == "global_steps" else "Epoch")
        ax.set_ylabel(f"val/{metric_name}")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
        ax.xaxis.set_major_formatter(x_formatter)
        ax.set_title(
            _format_subplot_title(
                label,
                max_total_chars=max_title_length,
                wrap_width=title_wrap_width,
                max_lines=title_max_lines,
            ),
            fontsize=10,
            linespacing=1.2,
            pad=8,
        )

        if y_limits is not None:
            ax.set_ylim(*y_limits)

    for index in range(n_runs, len(flat_axes)):
        flat_axes[index].axis("off")

    fig.suptitle(f"val/{metric_name} across runs", fontsize=14)
    fig.subplots_adjust(top=0.86, wspace=wspace, hspace=hspace)
    return fig
