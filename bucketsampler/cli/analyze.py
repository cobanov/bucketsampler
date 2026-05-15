"""Dataset analyzer command.

``bucketsampler analyze <path>`` walks an image directory, reads each
image's header (no decode), and reports how the dataset interacts with a
chosen bucket set: AR distribution, per-bucket counts, underutilized
buckets, outliers, and crop-loss summary. Three output formats are
supported: text (default, human-readable), JSON (for piping), and HTML
(matplotlib-rendered plots embedded as base64).

The scanner is I/O bound; header reads run on a thread pool sized via
``--workers``. Broken or unreadable images are counted and reported
separately, not skipped silently.
"""

from __future__ import annotations

import base64
import io
import json
import math
import sys
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import typer
from PIL import Image, UnidentifiedImageError

from bucketsampler.core.assignment import assign_many_indices, log_ar_distance
from bucketsampler.core.bucket import Bucket, BucketSet
from bucketsampler.core.stats import (
    aspect_ratio_summary,
    bucket_distribution,
    crop_loss_summary,
)
from bucketsampler.presets import list_presets, load_from_json, load_from_toml, load_preset

DEFAULT_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


class BucketCount(TypedDict):
    """One row of the per-bucket count table."""

    bucket: str
    width: int
    height: int
    count: int
    share: float


@dataclass(frozen=True, slots=True)
class OutlierImage:
    """One image whose AR is far from any bucket."""

    path: str
    width: int
    height: int
    aspect_ratio: float
    nearest_bucket: str
    log_ar_distance: float


@dataclass(frozen=True, slots=True)
class AnalyzeReport:
    """Structured analyzer output. Serializable via :func:`as_dict`."""

    root: str
    total_scanned: int
    readable: int
    broken: int
    broken_paths: list[str]
    bucket_set_name: str
    bucket_set_size: int
    vae_factor: int
    ar_summary: dict[str, float | int]
    crop_loss: dict[str, float | int]
    bucket_counts: list[BucketCount] = field(default_factory=list)
    underutilized: list[BucketCount] = field(default_factory=list)
    outliers: list[OutlierImage] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def scan_images(
    root: Path,
    *,
    suffixes: Iterable[str] = DEFAULT_SUFFIXES,
    recursive: bool = True,
) -> list[Path]:
    """Find image paths under ``root`` whose suffix matches ``suffixes``.

    Args:
        root: Directory to scan.
        suffixes: Lower-case file suffixes to include (``.jpg``, ``.png``, ...).
        recursive: If ``True``, descend into subdirectories.

    Returns:
        A sorted list of paths.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
        NotADirectoryError: If ``root`` is not a directory.
    """
    if not root.exists():
        raise FileNotFoundError(f"no such directory: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")
    suffix_set = {s.lower() for s in suffixes}
    iterator = root.rglob("*") if recursive else root.iterdir()
    paths = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in suffix_set)
    return paths


def read_dims_safe(paths: list[Path], *, workers: int = 8) -> tuple[np.ndarray, list[Path]]:
    """Read ``(width, height)`` for each path; return dims and broken paths.

    Header-only reads via PIL. Images that fail to open (truncated, wrong
    format, permissions) are collected into the broken list rather than
    aborting the scan.

    Args:
        paths: Image paths.
        workers: Threads. ``<= 1`` runs serially.

    Returns:
        Tuple ``(dims, broken)`` where ``dims`` is an ``(M, 2)`` int64 array
        for the M readable images and ``broken`` is the list of paths that
        failed to open.
    """

    def read_one(p: Path) -> tuple[Path, tuple[int, int] | None]:
        try:
            with Image.open(p) as img:
                return p, (int(img.width), int(img.height))
        except (UnidentifiedImageError, OSError, ValueError):
            return p, None

    if workers <= 1:
        results = [read_one(p) for p in paths]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(read_one, paths))

    dims: list[tuple[int, int]] = []
    broken: list[Path] = []
    for path, dim in results:
        if dim is None:
            broken.append(path)
        else:
            dims.append(dim)
    return np.asarray(dims, dtype=np.int64) if dims else np.zeros((0, 2), np.int64), broken


def build_report(
    *,
    root: Path,
    paths: list[Path],
    dims: np.ndarray,
    broken: list[Path],
    bucket_set: BucketSet,
    min_count: int = 10,
    outlier_threshold: float = 0.5,
    max_outliers_listed: int = 20,
) -> AnalyzeReport:
    """Run the analysis and produce a :class:`AnalyzeReport`.

    Args:
        root: Scanned directory (echoed in the report).
        paths: All readable image paths in scan order.
        dims: ``(M, 2)`` array of ``(width, height)`` for readable images.
        broken: Paths that failed to open.
        bucket_set: Buckets to evaluate against.
        min_count: Threshold below which a bucket is flagged as
            underutilized.
        outlier_threshold: Log-AR distance above which an image is flagged
            as an outlier.
        max_outliers_listed: Cap on listed outliers (the full count is
            still reported, only the worst N are emitted).
    """
    total_scanned = len(paths) + len(broken)
    ar_summary = aspect_ratio_summary(dims)
    crop_summary = crop_loss_summary(dims, bucket_set)
    counts = bucket_distribution(dims, bucket_set)
    bucket_counts: list[BucketCount] = [
        BucketCount(
            bucket=str(b),
            width=b.width,
            height=b.height,
            count=counts[b],
            share=counts[b] / max(len(paths), 1),
        )
        for b in bucket_set
    ]
    underutilized = [bc for bc in bucket_counts if bc["count"] < min_count]
    outliers = _find_outliers(
        paths=paths,
        dims=dims,
        bucket_set=bucket_set,
        threshold=outlier_threshold,
        max_listed=max_outliers_listed,
    )
    return AnalyzeReport(
        root=str(root),
        total_scanned=total_scanned,
        readable=len(paths),
        broken=len(broken),
        broken_paths=[str(p) for p in broken],
        bucket_set_name=bucket_set.name or "(unnamed)",
        bucket_set_size=len(bucket_set),
        vae_factor=bucket_set.vae_factor,
        ar_summary={
            "count": ar_summary.count,
            "min_ar": ar_summary.min_ar,
            "max_ar": ar_summary.max_ar,
            "mean_log_ar": ar_summary.mean_log_ar,
            "median_log_ar": ar_summary.median_log_ar,
            "std_log_ar": ar_summary.std_log_ar,
        },
        crop_loss={
            "count": crop_summary.count,
            "mean": crop_summary.mean,
            "median": crop_summary.median,
            "p95": crop_summary.p95,
            "max": crop_summary.max,
        },
        bucket_counts=bucket_counts,
        underutilized=underutilized,
        outliers=outliers,
    )


def _find_outliers(
    *,
    paths: list[Path],
    dims: np.ndarray,
    bucket_set: BucketSet,
    threshold: float,
    max_listed: int,
) -> list[OutlierImage]:
    if dims.size == 0:
        return []
    indices = assign_many_indices(dims, bucket_set)
    distances = np.empty(dims.shape[0], dtype=np.float64)
    nearest: list[Bucket] = []
    for i, (w, h) in enumerate(dims):
        b = bucket_set[int(indices[i])]
        nearest.append(b)
        distances[i] = log_ar_distance(int(w), int(h), b)
    mask = distances > threshold
    flagged = np.where(mask)[0]
    sorted_idx = flagged[np.argsort(-distances[flagged])]
    out: list[OutlierImage] = []
    for idx in sorted_idx[:max_listed]:
        w, h = int(dims[idx, 0]), int(dims[idx, 1])
        out.append(
            OutlierImage(
                path=str(paths[int(idx)]),
                width=w,
                height=h,
                aspect_ratio=w / h,
                nearest_bucket=str(nearest[int(idx)]),
                log_ar_distance=float(distances[int(idx)]),
            )
        )
    return out


def format_text(report: AnalyzeReport) -> str:
    """Render the report as a human-readable plain-text string."""
    lines: list[str] = []
    lines.append(f"Scanned {report.total_scanned} files under {report.root}")
    lines.append(f"  readable: {report.readable}")
    lines.append(f"  broken:   {report.broken}")
    if report.broken:
        for p in report.broken_paths[:5]:
            lines.append(f"    {p}")
        if report.broken > 5:
            lines.append(f"    ... and {report.broken - 5} more")
    lines.append("")
    s = report.ar_summary
    if s["count"]:
        lines.append("Aspect ratio (readable images):")
        lines.append(f"  min AR:         {s['min_ar']:.3f}")
        lines.append(f"  max AR:         {s['max_ar']:.3f}")
        lines.append(f"  mean log-AR:    {s['mean_log_ar']:.4f}")
        lines.append(f"  median log-AR:  {s['median_log_ar']:.4f}")
        lines.append(f"  std log-AR:     {s['std_log_ar']:.4f}")
        lines.append("")
    lines.append(
        f"Buckets ({report.bucket_set_name}, "
        f"{report.bucket_set_size} entries, vae_factor={report.vae_factor}):"
    )
    if report.bucket_counts:
        widest = max(bc["count"] for bc in report.bucket_counts) or 1
        for bc in report.bucket_counts:
            bar = "#" * round(20 * bc["count"] / widest) if widest else ""
            lines.append(
                f"  {bc['bucket']:<14} {bar:<20} {bc['count']:>6}  ({bc['share'] * 100:5.1f}%)"
            )
    lines.append("")
    if report.underutilized:
        lines.append(f"Underutilized buckets (< threshold): {len(report.underutilized)}")
        for bc in report.underutilized:
            lines.append(f"  {bc['bucket']:<14} {bc['count']}")
        lines.append("")
    if report.outliers:
        lines.append(f"Outliers (log-AR distance > threshold): listing top {len(report.outliers)}")
        for o in report.outliers:
            lines.append(
                f"  {o.path} ({o.width}x{o.height}, AR={o.aspect_ratio:.3f}, "
                f"nearest={o.nearest_bucket}, d={o.log_ar_distance:.3f})"
            )
        lines.append("")
    cl = report.crop_loss
    if cl["count"]:
        lines.append("Crop loss after fit-and-crop:")
        lines.append(f"  mean:   {cl['mean'] * 100:5.2f}%")
        lines.append(f"  median: {cl['median'] * 100:5.2f}%")
        lines.append(f"  p95:    {cl['p95'] * 100:5.2f}%")
        lines.append(f"  max:    {cl['max'] * 100:5.2f}%")
    return "\n".join(lines)


def format_json(report: AnalyzeReport) -> str:
    """Render the report as a JSON string with safe float handling.

    NaN and infinite floats (produced by summarizing an empty dataset) are
    rewritten to ``null`` so the output is valid JSON.
    """
    return json.dumps(_sanitize_for_json(report.as_dict()), indent=2)


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_for_json(v) for v in value]
    return value


def format_html(report: AnalyzeReport, *, dims: np.ndarray) -> str:
    """Render the report as a single self-contained HTML page.

    Plots are produced with matplotlib and inlined as base64 PNGs so the
    output is a single file with no external assets.

    Raises:
        ImportError: If matplotlib is not installed (install with the
            ``[analyze]`` extra).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "HTML report requires matplotlib. Install with: pip install bucketsampler[analyze]"
        ) from exc

    ar_plot = _plot_to_data_uri(plt, lambda ax: _draw_ar_histogram(ax, dims))
    bucket_plot = _plot_to_data_uri(plt, lambda ax: _draw_bucket_bar(ax, report.bucket_counts))
    text_body = format_text(report)
    return _HTML_TEMPLATE.format(
        title=f"bucketsampler analyze: {report.root}",
        ar_plot=ar_plot,
        bucket_plot=bucket_plot,
        text_body=_escape_html(text_body),
    )


def _plot_to_data_uri(plt_module: Any, draw: Any) -> str:
    fig, ax = plt_module.subplots(figsize=(7, 4), dpi=100)
    draw(ax)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt_module.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _draw_ar_histogram(ax: Any, dims: np.ndarray) -> None:
    if dims.size == 0:
        ax.set_title("Aspect ratio histogram (no data)")
        return
    log_ars = np.log(dims[:, 0] / dims[:, 1])
    ax.hist(log_ars, bins=40)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("log(width / height)")
    ax.set_ylabel("count")
    ax.set_title("Aspect ratio distribution")


def _draw_bucket_bar(ax: Any, counts: list[BucketCount]) -> None:
    if not counts:
        ax.set_title("Bucket distribution (no data)")
        return
    labels = [bc["bucket"] for bc in counts]
    values = [bc["count"] for bc in counts]
    ax.barh(labels, values)
    ax.invert_yaxis()
    ax.set_xlabel("images")
    ax.set_title("Per-bucket count")


def _escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
 body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 920px;
         margin: 2em auto; padding: 0 1em; color: #222; }}
 h1 {{ font-size: 1.4em; }}
 pre {{ background: #f4f4f4; padding: 1em; overflow-x: auto;
        border-radius: 4px; font-size: 13px; line-height: 1.4; }}
 img {{ max-width: 100%; height: auto; margin: 1em 0; }}
</style>
</head>
<body>
<h1>{title}</h1>
<img src="{ar_plot}" alt="aspect ratio histogram">
<img src="{bucket_plot}" alt="bucket distribution">
<pre>{text_body}</pre>
</body>
</html>
"""


def _load_bucket_set(preset: str | None, bucket_config: Path | None) -> BucketSet:
    if bool(preset) == bool(bucket_config):
        available = ", ".join(list_presets())
        raise typer.BadParameter(
            f"pass exactly one of --preset (available: {available}) or --bucket-config"
        )
    if preset:
        return load_preset(preset)
    assert bucket_config is not None
    if bucket_config.suffix.lower() == ".json":
        return load_from_json(bucket_config)
    return load_from_toml(bucket_config)


def analyze(
    path: Path = typer.Argument(..., exists=False, help="Image directory to scan."),
    preset: str | None = typer.Option(
        None,
        "--preset",
        "-p",
        help="Bundled preset name (sdxl, sd15, novelai).",
    ),
    bucket_config: Path | None = typer.Option(
        None,
        "--bucket-config",
        "-c",
        help="Path to a custom bucket TOML or JSON file.",
    ),
    recursive: bool = typer.Option(
        True, "--recursive/--no-recursive", help="Descend into subdirectories."
    ),
    workers: int = typer.Option(8, help="Threads for parallel header reads."),
    min_count: int = typer.Option(
        10, help="Buckets with fewer images than this are flagged as underutilized."
    ),
    outlier_threshold: float = typer.Option(
        0.5,
        help="Log-AR distance above which an image is flagged as an outlier.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit a JSON report instead of plain text."
    ),
    html_output: Path | None = typer.Option(
        None, "--html", help="Write a self-contained HTML report to this path."
    ),
) -> None:
    """Scan an image directory and report bucket statistics."""
    bucket_set = _load_bucket_set(preset, bucket_config)
    paths_found = scan_images(path, recursive=recursive)
    if not paths_found:
        typer.echo(f"No images found under {path}", err=True)
        raise typer.Exit(code=1)
    dims, broken = read_dims_safe(paths_found, workers=workers)
    readable_paths = [p for p in paths_found if p not in set(broken)]
    report = build_report(
        root=path,
        paths=readable_paths,
        dims=dims,
        broken=broken,
        bucket_set=bucket_set,
        min_count=min_count,
        outlier_threshold=outlier_threshold,
    )
    if html_output is not None:
        html_output.write_text(format_html(report, dims=dims), encoding="utf-8")
        typer.echo(f"Wrote {html_output}", err=True)
    if json_output:
        typer.echo(format_json(report))
    else:
        typer.echo(format_text(report))
    sys.stdout.flush()
