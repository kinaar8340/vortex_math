"""
Plotting and animation for vortex-math unit-circle visualizations.

Uses matplotlib for static plots / GIFs and plotly for interactive views.
High-resolution MP4 export can use NVIDIA NVENC (``h264_nvenc`` / ``hevc_nvenc``)
via ffmpeg when an RTX GPU is available — ideal on an RTX 4090.

Pure math lives in :mod:`core`; this module only renders.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Sequence  # Sequence used by modulus comparison

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from .core import (
    DEFAULT_LABEL_MODULUS,
    DEFAULT_STEP_RADIANS,
    SUGGESTED_MODULI,
    TRINITY_DIGITS,
    VORTEX_CYCLE,
    TWO_PI,
    circle_angles,
    circle_positions,
    digits_for_orbit,
    doubling_cycle_structure,
    doubling_edges,
    doubling_orbit,
    labels_for_orbit,
    modulus_sweep_report,
    vortex_number_circle_coords,
)

# ---------------------------------------------------------------------------
# Paths & style
# ---------------------------------------------------------------------------

_PKG_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ASSETS_DIR = _PKG_ROOT / "assets"

# Named resolutions (width, height) for high-res export.
RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "4k": (3840, 2160),
    "8k": (7680, 4320),
}

EncoderName = Literal[
    "auto",
    "h264_nvenc",
    "hevc_nvenc",
    "av1_nvenc",
    "libx264",
    "libx265",
]

# Practical max width/height per encoder (NVENC hardware limits).
# h264_nvenc on GeForce is capped near 4096×4096 — 8K (7680×4320) fails.
# hevc_nvenc / av1_nvenc on RTX 40-series support 8K.
_ENCODER_MAX_DIM: dict[str, int] = {
    "h264_nvenc": 4096,
    "hevc_nvenc": 8192,
    "av1_nvenc": 8192,
    "libx264": 16384,
    "libx265": 16384,
}


# ---------------------------------------------------------------------------
# FFmpeg / NVENC helpers (RTX 4090–friendly)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def ffmpeg_available() -> bool:
    """Return True if ``ffmpeg`` is on PATH."""
    return shutil.which("ffmpeg") is not None


@lru_cache(maxsize=1)
def list_ffmpeg_encoders() -> frozenset[str]:
    """Set of encoder names reported by ``ffmpeg -encoders``."""
    if not ffmpeg_available():
        return frozenset()
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
        )
    except (subprocess.SubprocessError, OSError):
        return frozenset()
    names: set[str] = set()
    for line in out.splitlines():
        # Lines look like: " V....D h264_nvenc   NVIDIA NVENC H.264 encoder"
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            names.add(parts[1])
    return frozenset(names)


def nvenc_available(codec: str = "h264_nvenc") -> bool:
    """True if ffmpeg lists the given NVENC encoder (e.g. on RTX 4090)."""
    return codec in list_ffmpeg_encoders()


def encoder_max_dimension(encoder: str) -> int:
    """Return a conservative max width/height for ``encoder``."""
    return _ENCODER_MAX_DIM.get(encoder, 8192)


def encoder_supports_resolution(
    encoder: str,
    width: int,
    height: int,
) -> bool:
    """Whether ``encoder`` can handle the given frame size."""
    limit = encoder_max_dimension(encoder)
    return max(width, height) <= limit and min(width, height) <= limit


def resolve_video_encoder(
    encoder: EncoderName | str = "auto",
    *,
    width: int | None = None,
    height: int | None = None,
    quiet: bool = False,
) -> str:
    """Pick a concrete ffmpeg video encoder, preferring NVENC when present.

    Preference order for ``auto`` (filtered by resolution when given):
      ``h264_nvenc`` → ``hevc_nvenc`` → ``av1_nvenc`` → ``libx265`` → ``libx264``

    For 8K frames, ``h264_nvenc`` is skipped automatically (hardware limit
    ~4096px). Explicit ``h264_nvenc`` at 8K is upgraded to ``hevc_nvenc``
    with a warning when available.
    """
    encoders = list_ffmpeg_encoders()
    w = width or 0
    h = height or 0
    need_check = w > 0 and h > 0

    def _log(msg: str) -> None:
        if not quiet:
            print(msg)

    def _ok(name: str) -> bool:
        if need_check and name in _ENCODER_MAX_DIM:
            if not encoder_supports_resolution(name, w, h):
                return False
        if name.endswith("_nvenc") and name not in encoders:
            return False
        if name.startswith("lib") and name not in encoders:
            return True  # try anyway
        return name in encoders or name.startswith("lib")

    if encoder == "auto":
        for candidate in (
            "h264_nvenc",
            "hevc_nvenc",
            "av1_nvenc",
            "libx265",
            "libx264",
        ):
            if _ok(candidate):
                if (
                    need_check
                    and candidate in {"hevc_nvenc", "av1_nvenc", "libx265"}
                    and not encoder_supports_resolution("h264_nvenc", w, h)
                ):
                    _log(
                        f"Note: {w}×{h} exceeds h264_nvenc limit "
                        f"(~{encoder_max_dimension('h264_nvenc')}px); "
                        f"using {candidate}"
                    )
                return candidate
        return "libx264"

    # Explicit encoder request
    if encoder.endswith("_nvenc") and encoder not in encoders:
        available = sorted(e for e in encoders if "nvenc" in e) or ["none"]
        _log(
            f"Warning: encoder {encoder!r} not found in ffmpeg; "
            f"falling back to auto (available NVENC: {available})"
        )
        return resolve_video_encoder(
            "auto", width=width, height=height, quiet=quiet
        )

    if need_check and not encoder_supports_resolution(encoder, w, h):
        _log(
            f"Warning: {encoder} cannot encode {w}×{h} "
            f"(max dim ~{encoder_max_dimension(encoder)}px). "
            f"Selecting a higher-tier encoder…"
        )
        for candidate in ("hevc_nvenc", "av1_nvenc", "libx265", "libx264"):
            if candidate == encoder:
                continue
            if _ok(candidate):
                _log(f"  → using {candidate}")
                return candidate
        raise RuntimeError(
            f"No encoder available for {w}×{h}. "
            f"Tried upgrading from {encoder!r}. "
            f"Install ffmpeg with hevc_nvenc (RTX) or libx265."
        )

    return encoder


def nvenc_ffmpeg_args(
    encoder: str = "h264_nvenc",
    cq: int = 18,
    preset: str = "p7",
    *,
    include_codec: bool = False,
) -> list[str]:
    """Extra ffmpeg args for high-quality NVENC (or software) encoding.

    Parameters
    ----------
    encoder :
        e.g. ``h264_nvenc``, ``hevc_nvenc``, ``av1_nvenc``, ``libx264``.
    cq :
        Constant quality target (lower = better; ~15–20 is excellent).
    preset :
        NVENC quality preset (``p1`` fastest … ``p7`` highest quality).
        For libx264 this maps to ``slow`` / ``medium``.
    include_codec :
        If True, also emit ``-c:v <encoder>``. Prefer setting the codec via
        Matplotlib's ``FFMpegWriter(codec=...)`` instead, so ffmpeg is not
        given conflicting ``-vcodec h264`` + ``-c:v h264_nvenc`` flags.
    """
    args: list[str] = ["-pix_fmt", "yuv420p"]
    if include_codec:
        args.extend(["-c:v", encoder])

    if encoder.endswith("_nvenc"):
        args.extend(
            [
                "-preset",
                preset,
                "-rc",
                "vbr",
                "-cq",
                str(cq),
                "-b:v",
                "0",
            ]
        )
        # Better compatibility in QuickTime / some players
        if encoder == "hevc_nvenc":
            args.extend(["-tag:v", "hvc1"])
    elif encoder in {"libx264", "libx265"}:
        x_preset = "slow" if preset in {"p6", "p7"} else "medium"
        args.extend(["-preset", x_preset, "-crf", str(cq)])
    return args


def parse_resolution(
    resolution: str | tuple[int, int] | Sequence[int] | None,
) -> tuple[int, int] | None:
    """Parse ``'4k'``, ``'1920x1080'``, or ``(w, h)`` into a pixel pair."""
    if resolution is None:
        return None
    if isinstance(resolution, str):
        key = resolution.strip().lower()
        if key in RESOLUTION_PRESETS:
            return RESOLUTION_PRESETS[key]
        if "x" in key:
            w_s, h_s = key.lower().split("x", 1)
            return int(w_s), int(h_s)
        raise ValueError(
            f"Unknown resolution {resolution!r}. "
            f"Use a preset {list(RESOLUTION_PRESETS)} or 'WIDTHxHEIGHT'."
        )
    if len(resolution) != 2:
        raise ValueError("resolution must be (width, height)")
    return int(resolution[0]), int(resolution[1])


def _fallback_encoder_chain(
    primary: str,
    width: int | None,
    height: int | None,
) -> list[str]:
    """Ordered list of encoders to try, starting with ``primary``."""
    chain = [primary]
    for candidate in (
        "hevc_nvenc",
        "av1_nvenc",
        "h264_nvenc",
        "libx265",
        "libx264",
    ):
        if candidate in chain:
            continue
        if width and height and not encoder_supports_resolution(candidate, width, height):
            continue
        if candidate.endswith("_nvenc") and candidate not in list_ffmpeg_encoders():
            continue
        chain.append(candidate)
    return chain


def save_animation(
    anim: animation.FuncAnimation,
    save_path: str | Path,
    *,
    fps: int = 30,
    dpi: int = 100,
    encoder: EncoderName | str = "auto",
    cq: int = 18,
    nvenc_preset: str = "p7",
    width: int | None = None,
    height: int | None = None,
) -> Path:
    """Save a Matplotlib animation as GIF or GPU-accelerated MP4.

    For ``.mp4`` / ``.mkv`` / ``.mov``: uses ffmpeg. When an NVIDIA GPU
    is available, prefers NVENC. **8K requires HEVC or AV1 NVENC** (or
    software ``libx265``) — ``h264_nvenc`` is limited to ~4K.

    For ``.gif``: uses Pillow (CPU).

    If the first encoder fails (e.g. resolution too large), automatically
    retries with the next capable encoder in the fallback chain.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".gif":
        anim.save(path, writer="pillow", fps=fps, dpi=dpi)
        print(f"Saved GIF → {path}")
        return path

    if suffix not in {".mp4", ".mkv", ".mov", ".webm"}:
        print(f"Note: unusual suffix {suffix!r}; encoding with ffmpeg")

    if not ffmpeg_available():
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg to save MP4 "
            "(with NVENC for GPU encode on RTX cards)."
        )

    # Infer pixel size from the figure if not provided
    if width is None or height is None:
        fig = anim._fig  # type: ignore[attr-defined]
        fig_w, fig_h = fig.get_size_inches()
        width = width or int(round(fig_w * dpi))
        height = height or int(round(fig_h * dpi))

    # yuv420p requires even dimensions
    width = int(width) - (int(width) % 2)
    height = int(height) - (int(height) % 2)

    primary = resolve_video_encoder(encoder, width=width, height=height)
    chain = _fallback_encoder_chain(primary, width, height)

    last_err: BaseException | None = None
    for attempt, resolved in enumerate(chain):
        extra = nvenc_ffmpeg_args(
            resolved, cq=cq, preset=nvenc_preset, include_codec=False
        )
        # Set codec on the writer so Matplotlib does not force ``-vcodec h264``
        writer = animation.FFMpegWriter(
            fps=fps,
            codec=resolved,
            bitrate=-1,
            extra_args=extra,
        )
        label = f"encoder={resolved}  {width}×{height}  fps={fps}  dpi={dpi}"
        if attempt == 0:
            print(f"Encoding {path.name} with {label}")
        else:
            print(f"Retrying {path.name} with {label}")

        try:
            anim.save(path, writer=writer, dpi=dpi)
            print(f"Saved GPU/high-quality animation → {path}  ({resolved})")
            return path
        except (subprocess.CalledProcessError, OSError, BrokenPipeError) as exc:
            last_err = exc
            print(
                f"  ! encode failed with {resolved}: "
                f"{type(exc).__name__}: {exc}"
            )
            # Remove empty / partial output before retry
            if path.exists() and path.stat().st_size == 0:
                path.unlink(missing_ok=True)
            continue

    raise RuntimeError(
        f"Failed to encode {path} at {width}×{height}. "
        f"Tried: {chain}. Last error: {last_err}\n"
        f"Hint: 8K needs hevc_nvenc/av1_nvenc (not h264_nvenc). "
        f"Check: ffmpeg -encoders | grep nvenc"
    ) from last_err

# Custom 9-color palette (distinct digits 1–9); viridis-adjacent but categorical.
VORTEX_PALETTE_9 = [
    "#440154",  # 1 deep purple
    "#482878",  # 2
    "#3e4a89",  # 3 (trinity-tinted cooler)
    "#31688e",  # 4
    "#26828e",  # 5
    "#1f9e89",  # 6
    "#35b779",  # 7
    "#6dcd59",  # 8
    "#fde725",  # 9 gold/highlight
]

# Highlight 3-6-9 with warmer accents when desired
TRINITY_HIGHLIGHT = {3: "#ff6b6b", 6: "#ffa94d", 9: "#ffe066"}


def _ensure_assets_dir(path: str | Path | None = None) -> Path:
    d = Path(path) if path is not None else DEFAULT_ASSETS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def apply_style(style: Literal["dark", "light"] = "dark") -> dict[str, Any]:
    """Return rcParams-friendly kwargs and set a consistent look."""
    if style == "dark":
        bg, fg, grid = "#0d1117", "#e6edf3", "#30363d"
        face = "#161b22"
    else:
        bg, fg, grid = "#ffffff", "#1a1a2e", "#dddddd"
        face = "#f8f9fa"
    plt.rcParams.update(
        {
            "figure.facecolor": bg,
            "axes.facecolor": face,
            "axes.edgecolor": grid,
            "axes.labelcolor": fg,
            "text.color": fg,
            "xtick.color": fg,
            "ytick.color": fg,
            "grid.color": grid,
            "savefig.facecolor": bg,
            "savefig.edgecolor": bg,
        }
    )
    return {"bg": bg, "fg": fg, "grid": grid, "face": face}


def vortex_colormap() -> ListedColormap:
    """Listed colormap for digits 1–9 (index 0 unused)."""
    # Boundaries: color digit d with palette[d-1]
    colors = ["#000000"] + VORTEX_PALETTE_9  # index 0 dummy
    return ListedColormap(VORTEX_PALETTE_9)


def label_colormap(modulus: int = DEFAULT_LABEL_MODULUS):
    """Colormap suited to labeling modulus ``m`` (categorical 9 vs continuous)."""
    if modulus == 9:
        return vortex_colormap()
    # Continuous map works for large m (37, 111, 333).
    return plt.get_cmap("turbo")


def _label_vmin_vmax(labels: np.ndarray, modulus: int) -> tuple[float, float]:
    """Color limits for scatter / colorbar given classic (1–9) vs modular (0..m-1)."""
    if modulus == 9 and labels.size and int(labels.min()) >= 1:
        return 0.5, 9.5
    return -0.5, float(modulus) - 0.5


def _digit_colors(digits: np.ndarray, modulus: int = DEFAULT_LABEL_MODULUS) -> np.ndarray:
    """Map label array to RGBA."""
    cmap = label_colormap(modulus)
    vmin, vmax = _label_vmin_vmax(digits, modulus)
    span = max(vmax - vmin, 1e-9)
    normed = (digits.astype(float) - vmin) / span
    return cmap(np.clip(normed, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Static plots
# ---------------------------------------------------------------------------


def plot_unit_circle_with_steps(
    num_steps: int = 50,
    step: float = DEFAULT_STEP_RADIANS,
    method: str = "step_index",
    modulus: int = DEFAULT_LABEL_MODULUS,
    style: Literal["dark", "light"] = "dark",
    label_first: int = 12,
    show_vortex_overlay: bool | None = None,
    show_doubling_star: bool = True,
    title: str | None = None,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (9, 9),
) -> Figure:
    """Scatter unit-circle points stepped by ``step`` radians, colored by label.

    Geometry defaults to ``9/π``. ``modulus`` only changes the discrete
    coloring / labeling (not the arc step). Classic 1–9 overlay is shown
    when ``modulus == 9`` unless overridden.

    Parameters
    ----------
    num_steps :
        Number of orbit points.
    step :
        Arc step in radians (default ``9/π``).
    method :
        Mapping method for :func:`core.position_to_label`.
    modulus :
        Labeling modulus (independent of ``step``).
    style :
        ``"dark"`` or ``"light"``.
    label_first :
        Annotate the first N points with step index.
    show_vortex_overlay :
        Draw the 1–9 number circle. Default: True only when ``modulus==9``.
    show_doubling_star :
        Connect the doubling circuit on the number circle.
    title :
        Optional figure title.
    ax :
        Existing axes; if None, a new figure is created.
    save_path :
        If set, save the figure to this path.
    figsize :
        Figure size when creating a new figure.

    Returns
    -------
    Figure
    """
    if show_vortex_overlay is None:
        show_vortex_overlay = modulus == 9

    colors_meta = apply_style(style)
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x, y = circle_positions(num_steps, step)
    digits = labels_for_orbit(num_steps, step, method=method, modulus=modulus)
    point_colors = _digit_colors(digits, modulus=modulus)

    # Unit circle guide
    theta = np.linspace(0, TWO_PI, 400)
    ax.plot(
        np.cos(theta),
        np.sin(theta),
        color=colors_meta["grid"],
        lw=1.0,
        alpha=0.7,
        zorder=1,
    )

    # Stepped points
    sizes = np.full(num_steps, 36.0)
    # Emphasize 3-6-9 slightly larger (only meaningful for classic digits)
    if modulus == 9:
        for t in TRINITY_DIGITS:
            sizes[digits == t] = 55.0

    vmin, vmax = _label_vmin_vmax(digits, modulus)
    sc = ax.scatter(
        x,
        y,
        c=digits,
        cmap=label_colormap(modulus),
        vmin=vmin,
        vmax=vmax,
        s=sizes,
        edgecolors=colors_meta["fg"],
        linewidths=0.35,
        alpha=0.9,
        zorder=5,
    )
    if modulus == 9:
        cbar = fig.colorbar(sc, ax=ax, ticks=range(1, 10), fraction=0.046, pad=0.04)
        cbar.set_label("Vortex digit")
    else:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"label mod {modulus}")

    # Labels for first few points
    n_lab = min(label_first, num_steps)
    for i in range(n_lab):
        ax.annotate(
            str(i),
            (x[i], y[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
            alpha=0.85,
            color=colors_meta["fg"],
            zorder=6,
        )

    if show_vortex_overlay:
        _draw_vortex_number_overlay(
            ax,
            radius=1.25,
            show_doubling_star=show_doubling_star,
            fg=colors_meta["fg"],
            grid=colors_meta["grid"],
        )

    ax.set_aspect("equal")
    pad = 1.55 if show_vortex_overlay else 1.15
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    ax.axhline(0, color=colors_meta["grid"], lw=0.5, alpha=0.5)
    ax.axvline(0, color=colors_meta["grid"], lw=0.5, alpha=0.5)
    ax.grid(True, alpha=0.25)

    if title is None:
        title = (
            f"Unit circle · step = 9/π ≈ {step:.5f} rad · "
            f"n = {num_steps} · map = {method} · m = {modulus}"
        )
    ax.set_title(title, fontsize=11, pad=12)
    ax.set_xlabel("x = cos(θ)")
    ax.set_ylabel("y = sin(θ)")

    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def _draw_vortex_number_overlay(
    ax: plt.Axes,
    radius: float = 1.25,
    show_doubling_star: bool = True,
    fg: str = "#e6edf3",
    grid: str = "#30363d",
) -> None:
    """Draw classic 1–9 number circle + doubling star outside/around unit circle."""
    coords = vortex_number_circle_coords(radius=radius)
    # Rim circle for the number diagram
    circ = Circle(
        (0, 0),
        radius,
        fill=False,
        linestyle="--",
        linewidth=0.9,
        edgecolor=grid,
        alpha=0.6,
        zorder=2,
    )
    ax.add_patch(circ)

    # Digits 1–9 on rim
    for digit, (cx, cy) in coords.items():
        is_trinity = digit in TRINITY_DIGITS
        is_circuit = digit in VORTEX_CYCLE
        color = TRINITY_HIGHLIGHT.get(digit, fg) if is_trinity else fg
        weight = "bold" if is_trinity or is_circuit else "normal"
        ax.plot(cx, cy, "o", ms=6 if is_trinity else 4, color=color, zorder=4, alpha=0.9)
        ax.text(
            cx * 1.08,
            cy * 1.08,
            str(digit),
            ha="center",
            va="center",
            fontsize=9,
            fontweight=weight,
            color=color,
            zorder=4,
        )

    # Special marker for 9 at center (control / completion)
    ax.plot(0, 0, marker="*", ms=14, color=TRINITY_HIGHLIGHT[9], zorder=7, alpha=0.95)
    ax.text(
        0,
        -0.12,
        "9",
        ha="center",
        va="top",
        fontsize=8,
        color=TRINITY_HIGHLIGHT[9],
        fontweight="bold",
        zorder=7,
    )

    if show_doubling_star:
        for a, b in doubling_edges():
            x0, y0 = coords[a]
            x1, y1 = coords[b]
            ax.plot(
                [x0, x1],
                [y0, y1],
                color="#58a6ff",
                lw=1.4,
                alpha=0.75,
                zorder=3,
            )


def plot_vortex_flow_on_circle(
    num_steps: int = 80,
    step: float = DEFAULT_STEP_RADIANS,
    method: str = "step_index",
    modulus: int = DEFAULT_LABEL_MODULUS,
    style: Literal["dark", "light"] = "dark",
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (12, 6),
) -> Figure:
    """Show vortex star / infinity pattern beside (or with) 9/π stepped points.

    Left panel: classic number circle + doubling star (and optional 3-6-9 hints).
    Right panel: unit-circle orbit colored by vortex digit, with faint path polyline.
    """
    colors_meta = apply_style(style)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize)

    # --- Left: pure vortex diagram ---
    coords = vortex_number_circle_coords(radius=1.0)
    theta = np.linspace(0, TWO_PI, 400)
    ax_left.plot(np.cos(theta), np.sin(theta), color=colors_meta["grid"], lw=1.2)
    for digit, (cx, cy) in coords.items():
        color = TRINITY_HIGHLIGHT.get(digit, colors_meta["fg"])
        ax_left.plot(cx, cy, "o", ms=10, color=color)
        ax_left.text(
            cx * 1.18,
            cy * 1.18,
            str(digit),
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=color,
        )
    for a, b in doubling_edges():
        x0, y0 = coords[a]
        x1, y1 = coords[b]
        ax_left.plot([x0, x1], [y0, y1], color="#58a6ff", lw=2.0, alpha=0.85)
    # Infinity-like hint: also draw 1-4-7 and 2-5-8 families sometimes shown
    # as secondary structure (light)
    for triple in ((1, 4, 7), (2, 5, 8)):
        pts = [coords[d] for d in triple]
        xs = [p[0] for p in pts] + [pts[0][0]]
        ys = [p[1] for p in pts] + [pts[0][1]]
        ax_left.plot(xs, ys, color="#f778ba", lw=0.9, alpha=0.35, ls=":")
    ax_left.plot(0, 0, marker="*", ms=16, color=TRINITY_HIGHLIGHT[9])
    ax_left.set_aspect("equal")
    ax_left.set_xlim(-1.5, 1.5)
    ax_left.set_ylim(-1.5, 1.5)
    ax_left.set_title("Vortex flow (doubling star 1-2-4-8-7-5)")
    ax_left.axis("off")

    # --- Right: stepped orbit (geometry 9/π; color by labeling modulus) ---
    x, y = circle_positions(num_steps, step)
    digits = labels_for_orbit(num_steps, step, method=method, modulus=modulus)
    ax_right.plot(np.cos(theta), np.sin(theta), color=colors_meta["grid"], lw=1.0, alpha=0.7)
    # Path polyline
    ax_right.plot(x, y, color=colors_meta["grid"], lw=0.6, alpha=0.35, zorder=2)
    vmin, vmax = _label_vmin_vmax(digits, modulus)
    sc = ax_right.scatter(
        x,
        y,
        c=digits,
        cmap=label_colormap(modulus),
        vmin=vmin,
        vmax=vmax,
        s=40,
        edgecolors=colors_meta["fg"],
        linewidths=0.3,
        zorder=5,
    )
    if modulus == 9:
        fig.colorbar(sc, ax=ax_right, ticks=range(1, 10), fraction=0.046, pad=0.04)
    else:
        fig.colorbar(sc, ax=ax_right, fraction=0.046, pad=0.04)
    ax_right.set_aspect("equal")
    ax_right.set_xlim(-1.2, 1.2)
    ax_right.set_ylim(-1.2, 1.2)
    ax_right.set_title(f"9/π steps (n={num_steps}, m={modulus})")
    ax_right.set_xlabel("x")
    ax_right.set_ylabel("y")
    ax_right.grid(True, alpha=0.25)

    fig.suptitle(
        "Vortex math on the unit circle · doubling circuit & irrational rotation",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def plot_density_heatmap(
    num_steps: int = 2000,
    step: float = DEFAULT_STEP_RADIANS,
    n_bins: int = 90,
    style: Literal["dark", "light"] = "dark",
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (8, 4),
) -> Figure:
    """Angular density of the stepped orbit (irrational rotation fill)."""
    colors_meta = apply_style(style)
    angles = circle_angles(num_steps, step)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.1, 1]})

    # Polar-style scatter density via 2D histogram on embedding
    x, y = np.cos(angles), np.sin(angles)
    h = ax0.hist2d(x, y, bins=n_bins, range=[[-1.05, 1.05], [-1.05, 1.05]], cmap="magma")
    fig.colorbar(h[3], ax=ax0, fraction=0.046, pad=0.04, label="count")
    th = np.linspace(0, TWO_PI, 300)
    ax0.plot(np.cos(th), np.sin(th), color=colors_meta["fg"], lw=0.8, alpha=0.5)
    ax0.set_aspect("equal")
    ax0.set_title(f"Plane density (n={num_steps})")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")

    # Angular histogram
    ax1.hist(
        angles,
        bins=n_bins,
        range=(0, TWO_PI),
        color="#58a6ff",
        edgecolor=colors_meta["grid"],
        alpha=0.85,
    )
    ax1.set_xlabel("θ (rad)")
    ax1.set_ylabel("count")
    ax1.set_title("Angular histogram")
    ax1.set_xlim(0, TWO_PI)

    fig.suptitle(f"Fill density · step ≈ {step:.5f} rad", fontsize=11)
    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def animate_circle_steps(
    num_steps: int = 200,
    step: float = DEFAULT_STEP_RADIANS,
    method: str = "step_index",
    modulus: int = DEFAULT_LABEL_MODULUS,
    interval: int = 50,
    style: Literal["dark", "light"] = "dark",
    save_gif: bool = False,
    save_mp4: bool = False,
    save_path: str | Path | None = None,
    assets_dir: str | Path | None = None,
    figsize: tuple[float, float] = (8, 8),
    blit: bool = False,
) -> animation.FuncAnimation:
    """Progressively draw stepped unit-circle points as an animation.

    Parameters
    ----------
    num_steps, step, method, modulus, interval :
        Orbit geometry (default step 9/π), labeling modulus, and animation
        timing (ms between frames).
    save_gif / save_mp4 :
        Persist animation under ``assets/`` (or ``save_path``).
    blit :
        Matplotlib blitting (False is safer across backends).

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    colors_meta = apply_style(style)
    x, y = circle_positions(num_steps, step)
    digits = labels_for_orbit(num_steps, step, method=method, modulus=modulus)
    rgba = _digit_colors(digits, modulus=modulus)

    fig, ax = plt.subplots(figsize=figsize)
    th = np.linspace(0, TWO_PI, 400)
    ax.plot(np.cos(th), np.sin(th), color=colors_meta["grid"], lw=1.0, alpha=0.7)
    _draw_vortex_number_overlay(
        ax, radius=1.25, show_doubling_star=True, fg=colors_meta["fg"], grid=colors_meta["grid"]
    )

    path_line, = ax.plot([], [], color=colors_meta["grid"], lw=0.7, alpha=0.4, zorder=2)
    scat = ax.scatter([], [], s=40, zorder=5)
    title = ax.set_title("")
    ax.set_aspect("equal")
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.55)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.2)

    def init():
        scat.set_offsets(np.empty((0, 2)))
        path_line.set_data([], [])
        title.set_text("")
        return scat, path_line, title

    def update(frame: int):
        n = frame + 1
        offsets = np.column_stack([x[:n], y[:n]])
        scat.set_offsets(offsets)
        scat.set_color(rgba[:n])
        path_line.set_data(x[:n], y[:n])
        title.set_text(
            f"Step {frame} · digit={digits[frame]} · θ={circle_angles(num_steps, step)[frame]:.3f}"
        )
        return scat, path_line, title

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=num_steps,
        interval=interval,
        blit=blit,
        repeat=True,
    )

    out_dir = _ensure_assets_dir(assets_dir)
    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        suffix = out.suffix.lower()
        if suffix == ".gif":
            anim.save(out, writer="pillow", fps=max(1, 1000 // interval))
        elif suffix in {".mp4", ".mkv"}:
            anim.save(out, writer="ffmpeg", fps=max(1, 1000 // interval))
        else:
            anim.save(out, writer="pillow", fps=max(1, 1000 // interval))
    else:
        if save_gif:
            gif_path = out_dir / "circle_steps.gif"
            try:
                anim.save(gif_path, writer="pillow", fps=max(1, 1000 // interval))
                print(f"Saved GIF → {gif_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"Could not save GIF: {exc}")
        if save_mp4:
            mp4_path = out_dir / "circle_steps.mp4"
            try:
                anim.save(mp4_path, writer="ffmpeg", fps=max(1, 1000 // interval))
                print(f"Saved MP4 → {mp4_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"Could not save MP4 (is ffmpeg installed?): {exc}")

    return anim


# ---------------------------------------------------------------------------
# Plotly interactive & 3D torus (bonus)
# ---------------------------------------------------------------------------


def plot_interactive_plotly(
    num_steps: int = 100,
    step: float = DEFAULT_STEP_RADIANS,
    method: str = "step_index",
    modulus: int = DEFAULT_LABEL_MODULUS,
    save_html: str | Path | None = None,
):
    """Interactive Plotly scatter with slider-friendly layout for steps.

    Note: full live sliders for recomputing orbits work best inside a
    notebook or Dash app; this function builds a rich static-interactive
    figure for a given ``num_steps`` / ``step``, and optionally frames
    for several step counts. Geometry stays at default 9/π; ``modulus``
    only changes label colors.
    """
    import plotly.graph_objects as go

    step_options = [20, 50, 100, 200, 400]
    if num_steps not in step_options:
        step_options.append(num_steps)
        step_options = sorted(set(step_options))

    cmin, cmax = (1, 9) if modulus == 9 else (0, modulus - 1)
    cbar_title = "digit" if modulus == 9 else f"mod {modulus}"

    frames = []
    for n in step_options:
        x, y = circle_positions(n, step)
        digits = labels_for_orbit(n, step, method=method, modulus=modulus)
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="markers+lines",
                        marker=dict(
                            size=8,
                            color=digits,
                            colorscale="Viridis",
                            cmin=cmin,
                            cmax=cmax,
                            colorbar=dict(title=cbar_title),
                            line=dict(width=0.5, color="#333"),
                        ),
                        line=dict(width=0.5, color="rgba(150,150,150,0.35)"),
                        text=[f"k={i}, d={d}" for i, d in enumerate(digits)],
                        hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>%{text}<extra></extra>",
                        name="orbit",
                    )
                ],
                name=str(n),
            )
        )

    # Initial data
    x0, y0 = circle_positions(num_steps, step)
    d0 = labels_for_orbit(num_steps, step, method=method, modulus=modulus)
    th = np.linspace(0, TWO_PI, 400)

    fig = go.Figure(
        data=[
            go.Scatter(
                x=np.cos(th),
                y=np.sin(th),
                mode="lines",
                line=dict(color="#888", width=1),
                name="unit circle",
                hoverinfo="skip",
            ),
            go.Scatter(
                x=x0,
                y=y0,
                mode="markers+lines",
                marker=dict(
                    size=8,
                    color=d0,
                    colorscale="Viridis",
                    cmin=cmin,
                    cmax=cmax,
                    colorbar=dict(title=cbar_title),
                ),
                line=dict(width=0.5, color="rgba(150,150,150,0.35)"),
                name="orbit",
            ),
        ],
        frames=frames,
    )

    # Vortex overlay points
    coords = vortex_number_circle_coords(radius=1.2)
    vx = [coords[d][0] for d in range(1, 10)]
    vy = [coords[d][1] for d in range(1, 10)]
    fig.add_trace(
        go.Scatter(
            x=vx,
            y=vy,
            mode="markers+text",
            text=[str(d) for d in range(1, 10)],
            textposition="top center",
            marker=dict(size=10, color="#ffd700"),
            name="vortex 1–9",
        )
    )
    # Doubling edges
    for a, b in doubling_edges():
        fig.add_trace(
            go.Scatter(
                x=[coords[a][0], coords[b][0]],
                y=[coords[a][1], coords[b][1]],
                mode="lines",
                line=dict(color="#58a6ff", width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=f"Interactive vortex circle · step=9/π · method={method} · m={modulus}",
        xaxis=dict(scaleanchor="y", scaleratio=1, range=[-1.6, 1.6], title="x"),
        yaxis=dict(range=[-1.6, 1.6], title="y"),
        template="plotly_dark",
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.12,
                x=0.0,
                xanchor="left",
                buttons=[
                    dict(
                        label="Play n-frames",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=600, redraw=True),
                                fromcurrent=True,
                            ),
                        ],
                    )
                ],
            )
        ],
        sliders=[
            dict(
                active=step_options.index(num_steps) if num_steps in step_options else 0,
                steps=[
                    dict(
                        label=str(n),
                        method="animate",
                        args=[[str(n)], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
                    )
                    for n in step_options
                ],
                x=0.1,
                len=0.8,
                currentvalue=dict(prefix="num_steps: "),
            )
        ],
        height=700,
    )

    if save_html is not None:
        path = Path(save_html)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path))
        print(f"Saved interactive HTML → {path}")
    return fig


def _torus_orbit_coords(
    num_steps: int,
    step: float,
    R: float,
    r: float,
    phi_scale: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parametric torus coordinates for the stepped 9/π orbit.

        x = (R + r cos φ) cos θ
        y = (R + r cos φ) sin θ
        z = r sin φ

    with ``θ_k = k · step`` and ``φ_k = θ_k · phi_scale`` (default 3 for
    a clear multi-winding pattern on the toroidal surface).
    """
    k = np.arange(num_steps, dtype=float)
    theta = k * step
    phi = theta * phi_scale
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return x, y, z


def _draw_torus_wireframe(
    ax,
    R: float,
    r: float,
    *,
    color: str = "black",
    linewidth: float = 0.6,
    alpha: float = 0.35,
    n_u: int = 40,
    n_v: int = 20,
    rstride: int = 2,
    cstride: int = 2,
):
    """Draw a torus wireframe mesh; return the Line3DCollection artist."""
    u = np.linspace(0, TWO_PI, n_u)
    v = np.linspace(0, TWO_PI, n_v)
    U, V = np.meshgrid(u, v)
    Tx = (R + r * np.cos(V)) * np.cos(U)
    Ty = (R + r * np.cos(V)) * np.sin(U)
    Tz = r * np.sin(V)
    return ax.plot_wireframe(
        Tx,
        Ty,
        Tz,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        rstride=rstride,
        cstride=cstride,
    )


def _set_wireframe_color(wire, color: str, alpha: float | None = None) -> None:
    """Update wireframe line color (Line3DCollection)."""
    if wire is None:
        return
    try:
        wire.set_color(color)
    except Exception:  # noqa: BLE001
        try:
            wire.set_edgecolor(color)
        except Exception:  # noqa: BLE001
            pass
    if alpha is not None:
        try:
            wire.set_alpha(alpha)
        except Exception:  # noqa: BLE001
            pass


def plot_torus_projection(
    num_steps: int = 300,
    step: float = DEFAULT_STEP_RADIANS,
    method: str = "step_index",
    modulus: int = DEFAULT_LABEL_MODULUS,
    R: float = 2.5,
    r: float = 1.0,
    phi_scale: float = 3.0,
    style: Literal["dark", "light"] = "dark",
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (12, 10),
    elev: float = 90.0,
    azim: float = 0.0,
) -> Figure:
    """Project stepped angles onto a torus (Rodin-coil inspired winding).

    High-contrast rendering with green wireframe option in the animation;
    static plot uses a **top-down** view (``elev=90``) looking along +Z
    so the doughnut hole faces the viewer.

    Parameters
    ----------
    num_steps :
        Number of orbit points.
    step :
        Arc step in radians (default ``9/π``).
    method :
        Label mapping for point colors.
    modulus :
        Labeling modulus (geometry step unchanged).
    R, r :
        Major / minor torus radii.
    phi_scale :
        Multiplier ``φ = θ · phi_scale`` controlling poloidal winding.
    elev, azim :
        Matplotlib 3D view angles (degrees). Default elev=90 is straight
        down the Z axis (face-on torus).
    """
    colors_meta = apply_style(style)
    x, y, z = _torus_orbit_coords(num_steps, step, R, r, phi_scale=phi_scale)
    digits = labels_for_orbit(num_steps, step, method=method, modulus=modulus)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Torus wireframe — black for high contrast against the grey pane
    wire_color = "black" if style == "dark" else "#1a1a1a"
    _draw_torus_wireframe(
        ax,
        R,
        r,
        color=wire_color,
        linewidth=0.65,
        alpha=0.32,
        n_u=40,
        n_v=20,
        rstride=2,
        cstride=2,
    )

    # Faint path so the winding is visible between points
    path_color = "#e6edf3" if style == "dark" else "#333333"
    ax.plot(x, y, z, color=path_color, linewidth=0.85, alpha=0.40, zorder=1)

    # Points: larger, white edge “glow”, colored by modular / vortex label
    vmin, vmax = _label_vmin_vmax(digits, modulus)
    scatter = ax.scatter(
        x,
        y,
        z,
        c=digits,
        cmap=label_colormap(modulus),
        vmin=vmin,
        vmax=vmax,
        s=38,
        alpha=0.95,
        edgecolors="white",
        linewidths=0.55,
        depthshade=True,
        zorder=5,
    )
    if modulus == 9:
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.08, ticks=range(1, 10))
        cbar.set_label("Vortex digit")
    else:
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.08)
        cbar.set_label(f"label mod {modulus}")

    ax.set_xlabel("X  (toroidal)", labelpad=8)
    ax.set_ylabel("Y  (toroidal)", labelpad=8)
    ax.set_zlabel("Z  (poloidal)", labelpad=8)
    ax.set_title(
        f"Toroidal projection of 9/π steps (n={num_steps}) — black wireframe\n"
        f"step ≈ {step:.5f} rad · φ = {phi_scale:g}·θ · map = {method} · m = {modulus}",
        pad=18,
        fontsize=12,
    )
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.3)
    # Equal-ish aspect so the torus is not stretched
    try:
        ax.set_box_aspect((1, 1, r / (R + r)))
    except Exception:  # noqa: BLE001 — older matplotlib
        pass

    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        print(f"Saved improved torus → {path}")
    return fig


def animate_torus_projection(
    num_steps: int = 300,
    step: float = DEFAULT_STEP_RADIANS,
    method: str = "step_index",
    modulus: int = DEFAULT_LABEL_MODULUS,
    R: float = 2.5,
    r: float = 1.0,
    phi_scale: float = 3.0,
    frames: int | None = None,
    fps: int = 30,
    interval: int | None = None,
    style: Literal["dark", "light"] = "dark",
    save_path: str | Path | None = None,
    elev: float = 90.0,
    elev_amp: float = 0.0,
    azim_start: float = 0.0,
    azim_sweep: float = 0.0,
    azim_step: float = 2.0,
    figsize: tuple[float, float] | None = None,
    resolution: str | tuple[int, int] | None = "1080p",
    dpi: int = 100,
    encoder: EncoderName | str = "auto",
    cq: int = 18,
    nvenc_preset: str = "p7",
    point_size: float | None = None,
    mode: Literal["construct", "spin"] = "construct",
    hold_final_frames: int = 15,
    rotate_seconds: float = 10.0,
    rotate_revolutions: float = 4.0,
    frames_per_step: int = 1,
) -> animation.FuncAnimation:
    """Animate the 9/π torus orbit — **build first**, then spin.

    Default mode is ``"construct"`` with a strict three-phase timeline:

    1. **Construct** — camera locked; one orbit step appears per frame
       (or ``frames_per_step`` frames). Length = ``num_steps * frames_per_step``.
       No rotation during this phase.
    2. **Hold** — full structure rests briefly (still no rotation).
    3. **Rotate** — only after the final step: spin for ``rotate_seconds``
       with increasing angular speed (θ ∝ t²), looking down +Z.

    Optional ``mode="spin"`` skips construction (full path from frame 0).

    Default export resolution is **1080p**. Encoding can use NVIDIA NVENC
    via ffmpeg on an RTX GPU.

    Parameters
    ----------
    num_steps, step, method, R, r, phi_scale :
        Orbit geometry and vortex-digit coloring (same as static torus plot).
    frames :
        Ignored in construct mode (timeline is derived from ``num_steps``).
        In ``spin`` mode: total frame count (default 180).
    frames_per_step :
        How many frames each new step stays visible during construction
        (default 1 → exactly ``--steps`` frames of build).
    fps :
        Output frames per second.
    mode :
        ``"construct"`` — build path, *then* accelerating spin (default).
        ``"spin"`` — full orbit already drawn; camera rotates.
    hold_final_frames :
        Pause after the last step before spin begins.
    rotate_seconds :
        Duration of the post-construction spin (default 10 s). Set ``0`` to skip.
    rotate_revolutions :
        Total turns over the spin phase with acceleration (θ ∝ t²).
    elev :
        Default ``90`` looks straight down +Z (face-on doughnut).
    resolution :
        Default ``"1080p"``.
    save_path :
        ``.mp4`` → ffmpeg (+ NVENC when available); ``.gif`` → Pillow.

    Examples
    --------
    >>> animate_torus_projection(  # doctest: +SKIP
    ...     num_steps=300,
    ...     resolution="1080p",
    ...     fps=30,
    ...     rotate_seconds=10,
    ...     save_path="assets/torus_1080p.mp4",
    ... )
    """
    apply_style(style)
    x, y, z = _torus_orbit_coords(num_steps, step, R, r, phi_scale=phi_scale)
    digits = labels_for_orbit(num_steps, step, method=method, modulus=modulus)
    rgba = _digit_colors(digits, modulus=modulus)

    # Strict timeline: construct ALL steps → hold → accelerate-rotate
    # Camera does not spin until construction (and hold) are finished.
    fps_i = max(1, int(fps))
    f_per_step = max(1, int(frames_per_step))
    if mode == "construct":
        build_frames = max(1, num_steps * f_per_step)
        hold_frames = max(0, int(hold_final_frames))
        rotate_frames = (
            max(0, int(round(fps_i * float(rotate_seconds))))
            if rotate_seconds > 0
            else 0
        )
        total_frames = build_frames + hold_frames + rotate_frames
        print(
            f"Timeline: construct {build_frames}f "
            f"({num_steps} steps × {f_per_step} f/step) → "
            f"hold {hold_frames}f → rotate {rotate_frames}f "
            f"({rotate_seconds:g}s, accelerating) · total {total_frames}f "
            f"@ {fps_i} fps ≈ {total_frames / fps_i:.1f}s"
        )
    else:
        build_frames = 0
        hold_frames = 0
        rotate_frames = max(1, int(frames) if frames is not None else 180)
        total_frames = rotate_frames
        print(f"Timeline: spin only · {total_frames}f @ {fps_i} fps")

    total_frames = max(1, total_frames)
    build_end = build_frames          # frames [0, build_end)
    hold_end = build_frames + hold_frames  # frames [build_end, hold_end)
    # rotate: frames [hold_end, total_frames)

    res = parse_resolution(resolution)
    if res is not None:
        width_px, height_px = res
        fig_w, fig_h = width_px / dpi, height_px / dpi
    elif figsize is not None:
        fig_w, fig_h = figsize
        width_px = int(fig_w * dpi)
        height_px = int(fig_h * dpi)
    else:
        width_px, height_px = RESOLUTION_PRESETS["1080p"]
        fig_w, fig_h = width_px / dpi, height_px / dpi

    if interval is None:
        interval = max(1, int(round(1000 / fps_i)))

    # Scale markers / wire density with resolution so 4K stays readable
    scale = max(1.0, width_px / 1920.0)
    if point_size is None:
        # ~10% of the previous default (36 → 3.6) so dots sit under path segments
        point_size = 3.6 * scale
    wire_lw = 0.55 + 0.15 * min(scale, 3.0)
    n_u = int(min(80, 40 + 10 * scale))
    n_v = int(min(50, 22 + 6 * scale))
    path_lw = 0.9 * scale

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    # Fixed axis limits so the view does not jump as points appear
    pad = 0.15 * (R + r)
    lim = R + r + pad
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-r - pad, r + pad)

    # Green while constructing; switches to black when structure is complete
    wire_color_build = "#00FF00"
    wire_color_done = "black"
    wire = _draw_torus_wireframe(
        ax,
        R,
        r,
        color=wire_color_build,
        linewidth=wire_lw,
        alpha=0.55,
        n_u=n_u,
        n_v=n_v,
        rstride=2,
        cstride=2,
    )

    path_color = "#e6edf3" if style == "dark" else "#333333"
    # Start empty — path and points grow during construct mode
    (path_line,) = ax.plot(
        [],
        [],
        [],
        color=path_color,
        linewidth=path_lw,
        alpha=0.45,
        zorder=1,
    )

    # Seed scatter with a single invisible-ish point so 3D offsets work
    scatter = ax.scatter(
        [x[0]],
        [y[0]],
        [z[0]],
        c=[digits[0]],
        cmap=vortex_colormap(),
        vmin=0.5,
        vmax=9.5,
        s=point_size,
        alpha=0.95,
        edgecolors="none",
        linewidths=0.0,
        depthshade=True,
        zorder=5,
    )
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.55, pad=0.08, ticks=range(1, 10))
    cbar.set_label("Vortex digit", fontsize=10 + 2 * min(scale, 2))

    res_label = f"{width_px}×{height_px}"
    enc_hint = (
        resolve_video_encoder(
            encoder, width=width_px, height=height_px, quiet=True
        )
        if save_path
        else "n/a"
    )
    title = ax.set_title("", pad=16, fontsize=11 + min(scale, 2))

    # No XYZ grid / panes — only the torus + constructing path
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    try:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        # Fully transparent pane edges (no grey “walls”)
        ax.xaxis.pane.set_edgecolor((1, 1, 1, 0))
        ax.yaxis.pane.set_edgecolor((1, 1, 1, 0))
        ax.zaxis.pane.set_edgecolor((1, 1, 1, 0))
        ax.xaxis.line.set_color((1, 1, 1, 0))
        ax.yaxis.line.set_color((1, 1, 1, 0))
        ax.zaxis.line.set_color((1, 1, 1, 0))
    except Exception:  # noqa: BLE001
        pass
    try:
        ax.set_axis_off()
    except Exception:  # noqa: BLE001
        pass
    try:
        ax.set_box_aspect((1, 1, r / (R + r)))
    except Exception:  # noqa: BLE001
        pass

    # Locked top-down camera for construct + hold (no spin until phase 3)
    locked_elev = float(elev)
    locked_azim = float(azim_start)
    ax.view_init(elev=locked_elev, azim=locked_azim)

    def _phase(frame: int) -> str:
        if mode != "construct":
            return "spin"
        if frame < build_end:
            return "construct"
        if frame < hold_end:
            return "hold"
        return "rotate"

    def _visible_count(frame: int) -> int:
        """How many orbit points are visible at this frame."""
        if mode == "spin":
            return num_steps
        phase = _phase(frame)
        if phase in {"hold", "rotate"}:
            return num_steps
        # Construct: exactly one new step every frames_per_step frames
        # frame 0..f_per_step-1 → 1 point, …, last build frames → num_steps
        step_idx = frame // f_per_step  # 0-based step index
        return max(1, min(num_steps, step_idx + 1))

    def _set_path(n: int) -> None:
        if n <= 0:
            path_line.set_data_3d([], [], [])
            scatter._offsets3d = (np.array([]), np.array([]), np.array([]))
            scatter.set_array(np.array([]))
            return
        xs, ys, zs = x[:n], y[:n], z[:n]
        path_line.set_data_3d(xs, ys, zs)
        scatter._offsets3d = (xs, ys, zs)
        scatter.set_array(digits[:n].astype(float))
        try:
            scatter.set_facecolors(rgba[:n])
            scatter.set_edgecolors("none")
        except Exception:  # noqa: BLE001
            pass
        scatter.set_sizes(np.full(n, point_size))

    def _apply_wire_color(*, phase: str, n_visible: int) -> None:
        """Green while building; black as soon as the final step is on screen."""
        complete = phase != "construct" or n_visible >= num_steps
        if complete:
            _set_wireframe_color(wire, wire_color_done, alpha=0.70)
        else:
            _set_wireframe_color(wire, wire_color_build, alpha=0.55)

    def init():
        if mode == "construct":
            _set_path(0)
            _apply_wire_color(phase="construct", n_visible=0)
            title.set_text(
                f"Phase 1/3 · Constructing · step 0/{num_steps} · {res_label}"
            )
        else:
            _set_path(num_steps)
            _apply_wire_color(phase="spin", n_visible=num_steps)
            title.set_text(
                f"Torus · 9/π vortex steps (n={num_steps}) · {res_label}"
            )
        ax.view_init(elev=locked_elev, azim=locked_azim)
        return scatter, path_line, title

    def update(frame: int):
        phase = _phase(frame)
        n = _visible_count(frame)
        _set_path(n)
        _apply_wire_color(phase=phase, n_visible=n)

        if phase == "construct":
            # Camera frozen — only the structure grows (green wireframe)
            ax.view_init(elev=locked_elev, azim=locked_azim)
            digit = int(digits[n - 1]) if n > 0 else 0
            title.set_text(
                f"Phase 1/3 · Constructing · step {n}/{num_steps}  "
                f"(digit {digit}) · {res_label}"
            )
        elif phase == "hold":
            # Final step reached — wireframe turns black
            ax.view_init(elev=locked_elev, azim=locked_azim)
            title.set_text(
                f"Phase 2/3 · Construction complete · {num_steps} steps · "
                f"starting rotation… · {res_label}"
            )
        elif phase == "rotate":
            # Spin starts ONLY after final step (+ hold); wireframe stays black
            rot_i = frame - hold_end  # 0 .. rotate_frames-1
            u = (rot_i + 1) / max(rotate_frames, 1)  # (0, 1]
            # θ ∝ u² → angular speed increases from 0 through the 10 s
            azim_t = locked_azim + 360.0 * rotate_revolutions * (u * u)
            ax.view_init(elev=locked_elev, azim=float(azim_t))
            t_sec = (rot_i + 1) / fps_i
            omega = (
                2.0 * 360.0 * rotate_revolutions * t_sec
                / max(float(rotate_seconds), 1e-9) ** 2
                if rotate_seconds > 0
                else 0.0
            )
            title.set_text(
                f"Phase 3/3 · Rotating · {t_sec:.1f}s / {rotate_seconds:g}s  "
                f"· speed ↑ {omega:.0f}°/s · {res_label}"
            )
        else:  # pure spin mode
            u = (frame + 1) / max(total_frames, 1)
            azim_t = locked_azim + 360.0 * max(rotate_revolutions, 1.0) * (u * u)
            if rotate_revolutions <= 0:
                azim_t = locked_azim + frame * azim_step
            elev_t = locked_elev + elev_amp * np.sin(frame / 20.0)
            ax.view_init(elev=float(elev_t), azim=float(azim_t))
            title.set_text(
                f"Torus · 9/π vortex steps (n={num_steps}) · {res_label}"
            )
        return scatter, path_line, title

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=total_frames,
        interval=interval,
        blit=False,
        repeat=False,
    )

    if save_path is not None:
        save_animation(
            anim,
            save_path,
            fps=fps_i,
            dpi=dpi,
            encoder=encoder,
            cq=cq,
            nvenc_preset=nvenc_preset,
            width=width_px,
            height=height_px,
        )

    return anim


def plot_modulus_comparison(
    moduli: Sequence[int] | None = None,
    num_steps: int = 120,
    step: float = DEFAULT_STEP_RADIANS,
    method: str = "step_index",
    style: Literal["dark", "light"] = "dark",
    save_path: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Side-by-side unit-circle orbits for several labeling moduli.

    Geometry is fixed (default ``9/π``). Only the discrete label set
    changes — the practical way to see mechanism swap for m ∈ {9, 37, 111, 333}.
    """
    if moduli is None:
        moduli = list(SUGGESTED_MODULI)
    n = len(moduli)
    if figsize is None:
        figsize = (4.2 * n, 4.4)

    colors_meta = apply_style(style)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes_row = axes[0]

    x, y = circle_positions(num_steps, step)
    th = np.linspace(0, TWO_PI, 400)
    cx, cy = np.cos(th), np.sin(th)

    for ax, m in zip(axes_row, moduli):
        labels = labels_for_orbit(num_steps, step, method=method, modulus=int(m))
        vmin, vmax = _label_vmin_vmax(labels, int(m))
        ax.plot(cx, cy, color=colors_meta["grid"], lw=0.9, alpha=0.7)
        sc = ax.scatter(
            x,
            y,
            c=labels,
            cmap=label_colormap(int(m)),
            vmin=vmin,
            vmax=vmax,
            s=22,
            edgecolors=colors_meta["fg"],
            linewidths=0.2,
            alpha=0.9,
            zorder=5,
        )
        info = doubling_cycle_structure(int(m))
        orbit_1, len_1 = doubling_orbit(1, int(m))
        ax.set_aspect("equal")
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_title(
            f"m = {m}\n×2 from 1: len {len_1} · cycles {info['num_cycles']}",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Labeling modulus sweep · geometric step fixed at 9/π · method={method} · n={num_steps}",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def generate_default_assets(assets_dir: str | Path | None = None) -> dict[str, Path]:
    """Generate the key static visualizations into ``assets/``."""
    out = _ensure_assets_dir(assets_dir)
    paths: dict[str, Path] = {}

    p1 = out / "unit_circle_steps.png"
    fig1 = plot_unit_circle_with_steps(num_steps=100, save_path=p1)
    plt.close(fig1)
    paths["unit_circle_steps"] = p1

    p2 = out / "vortex_flow.png"
    fig2 = plot_vortex_flow_on_circle(num_steps=80, save_path=p2)
    plt.close(fig2)
    paths["vortex_flow"] = p2

    p3 = out / "density_heatmap.png"
    fig3 = plot_density_heatmap(num_steps=2000, save_path=p3)
    plt.close(fig3)
    paths["density_heatmap"] = p3

    p4 = out / "torus_projection.png"
    fig4 = plot_torus_projection(num_steps=300, save_path=p4)
    plt.close(fig4)
    paths["torus_projection"] = p4

    p5 = out / "interactive_circle.html"
    plot_interactive_plotly(num_steps=100, save_html=p5)
    paths["interactive_html"] = p5

    return paths
