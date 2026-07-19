#!/usr/bin/env python3
"""
CLI entry point for vortex math unit-circle visualizations.

Examples
--------
    python src/main.py --plot-steps --num-steps 100
    python src/main.py --plot-steps -m 37 --step-mode m_over_pi
    python src/main.py --plot-steps -m 37 --method paired
    python src/main.py --sweep-moduli --step-mode m_over_pi --sweep-views circle,density,torus
    python src/main.py --sweep-moduli --extended
    python src/main.py --torus --steps 300
    python src/main.py --demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python src/main.py` without installing the package.
_SRC = Path(__file__).resolve().parent
_ROOT = _SRC.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

# Non-interactive backend when saving; switch if user wants to show windows.
if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.core import (
    DEFAULT_LABEL_MODULUS,
    DEFAULT_STEP_RADIANS,
    SUGGESTED_MODULI,
    digital_root,
    doubling_orbit,
    modulus_sweep_report,
    resolve_moduli,
    step_radians_for,
    vortex_doubling_sequence,
)
from src.visualize import (
    DEFAULT_ASSETS_DIR,
    animate_circle_steps,
    animate_torus_projection,
    generate_default_assets,
    plot_density_heatmap,
    plot_density_modulus_comparison,
    plot_interactive_plotly,
    plot_modulus_comparison,
    plot_paired_label_orbit,
    plot_torus_modulus_comparison,
    plot_torus_projection,
    plot_unit_circle_with_steps,
    plot_vortex_flow_on_circle,
)


def _parse_explicit_step(value: str | None) -> float | None:
    """Parse --step if given; None means 'use step_mode'."""
    if value is None:
        return None
    v = value.strip().lower().replace(" ", "")
    if v in {"9/pi", "9/π", "default"}:
        return DEFAULT_STEP_RADIANS
    if v in {"m/pi", "m/π"}:
        return None  # signal: use m_over_pi via step_mode
    return float(value)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Vortex math · unit circle stepped by 9/π radians",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--plot-steps",
        action="store_true",
        help="Scatter plot of stepped unit-circle points colored by vortex digit",
    )
    p.add_argument(
        "--vortex-flow",
        action="store_true",
        help="Side-by-side vortex star and stepped orbit",
    )
    p.add_argument(
        "--density",
        action="store_true",
        help="Angular / plane density of a long orbit",
    )
    p.add_argument(
        "--torus",
        action="store_true",
        help="3D toroidal projection of the stepped points",
    )
    p.add_argument(
        "--animate-torus",
        action="store_true",
        help="Animate torus construction (point-by-point 9/π winding; default 1080p MP4)",
    )
    p.add_argument(
        "--animate",
        action="store_true",
        help="Animate progressive drawing of circle steps",
    )
    p.add_argument(
        "--interactive",
        action="store_true",
        help="Build Plotly interactive HTML (saved to assets/)",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Generate the key static visualizations into assets/",
    )
    p.add_argument("--num-steps", "--steps", type=int, default=100, dest="num_steps")
    p.add_argument(
        "--step",
        type=str,
        default=None,
        help="Explicit step in radians (float or '9/pi'). Overrides step-mode when set.",
    )
    p.add_argument(
        "--step-mode",
        type=str,
        default="nine_over_pi",
        choices=["nine_over_pi", "m_over_pi", "9_over_pi"],
        help=(
            "Geometric step policy: nine_over_pi (default, decoupled) or "
            "m_over_pi (couple winding rate to --modulus)"
        ),
    )
    p.add_argument(
        "--method",
        type=str,
        default="step_index",
        choices=[
            "step_index",
            "angle_bin",
            "sin_dr",
            "cos_dr",
            "doubling_cycle",
            "mod",
            "paired",
        ],
        help="Angle/index → label mapping (paired = CRT digital_root × k%%m)",
    )
    p.add_argument(
        "--modulus",
        "-m",
        type=int,
        default=DEFAULT_LABEL_MODULUS,
        help=(
            "Labeling modulus (and m/π step when --step-mode m_over_pi). "
            "Try 7, 9, 13, 27, 37, 41, 111, 333."
        ),
    )
    p.add_argument(
        "--sweep-moduli",
        action="store_true",
        help=(
            "Print doubling-cycle structure and comparison plots for moduli "
            f"{list(SUGGESTED_MODULI)} (see --extended, --sweep-views)"
        ),
    )
    p.add_argument(
        "--extended",
        action="store_true",
        help="With --sweep-moduli: also include 7, 13, 27, 41",
    )
    p.add_argument(
        "--sweep-views",
        type=str,
        default="circle",
        help=(
            "Comma list of sweep figures: circle, density, torus, all. "
            "Example: circle,density,torus"
        ),
    )
    p.add_argument(
        "--paired-panel",
        action="store_true",
        help="Save dual-panel CRT plot (digital root | k%%m | packed) for --modulus",
    )
    p.add_argument(
        "--style",
        type=str,
        default="dark",
        choices=["dark", "light"],
    )
    p.add_argument("--save-gif", action="store_true", help="Save animation as GIF")
    p.add_argument("--save-mp4", action="store_true", help="Save animation as MP4")
    p.add_argument(
        "--interval",
        type=int,
        default=40,
        help="Animation frame interval in ms (circle anim; torus prefers --fps)",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for torus / video export",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Animation frames (default: num_steps+hold for construct, 180 for spin)",
    )
    p.add_argument(
        "--resolution",
        type=str,
        default="1080p",
        help="Video resolution: 720p, 1080p (default), 1440p, 4k, 8k, or WIDTHxHEIGHT",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="construct",
        choices=["construct", "spin"],
        help="Torus anim: construct (build path) or spin (rotate finished orbit)",
    )
    p.add_argument(
        "--4k",
        dest="uhd_4k",
        action="store_true",
        help="Shorthand for --resolution 4k --save-mp4 (NVENC when available)",
    )
    p.add_argument(
        "--8k",
        dest="uhd_8k",
        action="store_true",
        help="Shorthand for --resolution 8k --save-mp4 (heavy; NVENC recommended)",
    )
    p.add_argument(
        "--encoder",
        type=str,
        default="auto",
        help="Video encoder: auto (prefer NVENC), h264_nvenc, hevc_nvenc, av1_nvenc, libx264",
    )
    p.add_argument(
        "--cq",
        type=int,
        default=18,
        help="NVENC CQ / x264 CRF quality (lower = better; 15–20 is excellent)",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path for a single figure/animation",
    )
    p.add_argument(
        "--assets-dir",
        type=str,
        default=str(DEFAULT_ASSETS_DIR),
        help="Directory for generated assets",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively (needs GUI backend)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    assets = Path(args.assets_dir)
    assets.mkdir(parents=True, exist_ok=True)

    # If no action flags, run demo
    actions = any(
        [
            args.plot_steps,
            args.vortex_flow,
            args.density,
            args.torus,
            args.animate_torus,
            args.animate,
            args.interactive,
            args.demo,
            args.sweep_moduli,
            args.paired_panel,
        ]
    )
    if not actions:
        args.demo = True

    modulus = int(args.modulus)
    if modulus <= 0:
        print("error: --modulus must be positive", file=sys.stderr)
        return 2

    # Normalize step mode aliases
    step_mode = args.step_mode.replace("-", "_")
    if step_mode == "9_over_pi":
        step_mode = "nine_over_pi"

    explicit = _parse_explicit_step(args.step)
    if args.step is not None and args.step.strip().lower().replace(" ", "") in {
        "m/pi",
        "m/π",
    }:
        step_mode = "m_over_pi"
        explicit = None

    step = step_radians_for(
        modulus=modulus,
        step_mode=step_mode,
        explicit=explicit,
    )

    print("Vortex Math Unit Circle")
    print(f"  step_mode = {step_mode}")
    print(f"  step = {step:.8f} rad  (9/π ≈ {DEFAULT_STEP_RADIANS:.8f})")
    print(f"  label modulus m = {modulus}")
    if step_mode == "nine_over_pi" and explicit is None:
        print("  note: geometry decoupled from m (labels only)")
    elif step_mode == "m_over_pi":
        print(f"  note: geometry coupled — Δθ = m/π = {modulus}/π")
    print(f"  method = {args.method}")
    print(f"  doubling cycle sample (m=9): {vortex_doubling_sequence(12)}")
    orbit_m, len_m = doubling_orbit(1, modulus)
    print(f"  ×2 orbit from 1 mod {modulus}: len={len_m}  prefix={orbit_m[:12]}")
    print(f"  digital_root(247) = {digital_root(247)}")
    print(f"  assets → {assets.resolve()}")

    if args.sweep_moduli:
        moduli = resolve_moduli(extended=args.extended)
        print("\n[sweep-moduli] Doubling structure (×2 on Z/mZ):")
        print(f"  moduli = {list(moduli)}")
        print(f"  step_mode = {step_mode}")
        print(f"  {'m':>6}  {'len_from_1':>10}  {'num_cycles':>10}  {'cycle_lengths'}")
        for row in modulus_sweep_report(moduli):
            print(
                f"  {row['modulus']:>6}  {row['length_from_1']:>10}  "
                f"{row['num_cycles']:>10}  {row['cycle_lengths']}"
            )

        views_raw = args.sweep_views.strip().lower()
        if views_raw == "all":
            views = {"circle", "density", "torus"}
        else:
            views = {v.strip() for v in views_raw.split(",") if v.strip()}

        tag = "m_over_pi" if step_mode == "m_over_pi" else "fixed9"
        if args.extended:
            tag += "_ext"

        if "circle" in views:
            out = (
                Path(args.out)
                if args.out and len(views) == 1
                else assets / f"modulus_comparison_{tag}.png"
            )
            fig = plot_modulus_comparison(
                moduli=moduli,
                num_steps=max(args.num_steps, 100),
                step=step if step_mode == "nine_over_pi" else None,
                step_mode=step_mode,
                method=args.method,
                style=args.style,
                save_path=out,
            )
            print(f"[sweep-moduli] circle comparison → {out}")
            if args.show:
                plt.show()
            else:
                plt.close(fig)

        if "density" in views:
            out_d = assets / f"modulus_density_{tag}.png"
            fig_d = plot_density_modulus_comparison(
                moduli=moduli,
                num_steps=max(args.num_steps, 1500),
                step_mode=step_mode,
                style=args.style,
                save_path=out_d,
            )
            print(f"[sweep-moduli] density comparison → {out_d}")
            if args.show:
                plt.show()
            else:
                plt.close(fig_d)

        if "torus" in views:
            out_t = assets / f"modulus_torus_{tag}.png"
            fig_t = plot_torus_modulus_comparison(
                moduli=moduli,
                num_steps=max(args.num_steps, 150),
                step_mode=step_mode,
                method=args.method,
                style=args.style,
                save_path=out_t,
            )
            print(f"[sweep-moduli] torus comparison → {out_t}")
            if args.show:
                plt.show()
            else:
                plt.close(fig_t)

    if args.paired_panel:
        out_p = Path(args.out) if args.out else assets / f"paired_labels_m{modulus}.png"
        fig_p = plot_paired_label_orbit(
            num_steps=max(args.num_steps, 120),
            step=step,
            modulus=modulus,
            style=args.style,
            save_path=out_p,
        )
        print(f"[paired-panel] saved → {out_p}")
        if args.show:
            plt.show()
        else:
            plt.close(fig_p)

    if args.demo:
        print("\n[demo] Generating default assets…")
        paths = generate_default_assets(assets)
        for name, path in paths.items():
            print(f"  {name}: {path}")

    if args.plot_steps:
        if args.out:
            out = Path(args.out)
        else:
            parts = [f"m{modulus}"]
            if step_mode == "m_over_pi":
                parts.append("step_m_over_pi")
            if args.method != "step_index":
                parts.append(args.method)
            if modulus == DEFAULT_LABEL_MODULUS and step_mode == "nine_over_pi" and args.method == "step_index":
                out = assets / "unit_circle_steps.png"
            else:
                out = assets / f"unit_circle_steps_{'_'.join(parts)}.png"
        fig = plot_unit_circle_with_steps(
            num_steps=args.num_steps,
            step=step,
            method=args.method,
            modulus=modulus,
            style=args.style,
            save_path=out,
        )
        print(f"[plot-steps] saved → {out}")
        if args.show:
            plt.show()
        else:
            plt.close(fig)

    if args.vortex_flow:
        out = Path(args.out) if args.out else assets / "vortex_flow.png"
        fig = plot_vortex_flow_on_circle(
            num_steps=args.num_steps,
            step=step,
            method=args.method,
            modulus=modulus,
            style=args.style,
            save_path=out,
        )
        print(f"[vortex-flow] saved → {out}")
        if args.show:
            plt.show()
        else:
            plt.close(fig)

    if args.density:
        out = Path(args.out) if args.out else assets / "density_heatmap.png"
        fig = plot_density_heatmap(
            num_steps=max(args.num_steps, 500),
            step=step,
            style=args.style,
            save_path=out,
        )
        print(f"[density] saved → {out}")
        if args.show:
            plt.show()
        else:
            plt.close(fig)

    if args.torus:
        out = Path(args.out) if args.out else assets / f"torus_projection_m{modulus}.png"
        if args.out is None and modulus == DEFAULT_LABEL_MODULUS:
            out = assets / "torus_projection.png"
        fig = plot_torus_projection(
            num_steps=args.num_steps,
            step=step,
            method=args.method,
            modulus=modulus,
            style=args.style,
            save_path=out,
        )
        print(f"[torus] saved → {out}")
        if args.show:
            plt.show()
        else:
            plt.close(fig)

    if args.animate_torus:
        resolution = args.resolution
        if args.uhd_8k:
            resolution = "8k"
        elif args.uhd_4k:
            resolution = "4k"

        # Default is 1080p MP4; GIF only when explicitly requested
        want_mp4 = not args.save_gif or args.save_mp4 or args.uhd_4k or args.uhd_8k
        if args.save_gif and not (args.uhd_4k or args.uhd_8k or args.save_mp4):
            want_mp4 = False

        if args.out:
            out = Path(args.out)
        elif want_mp4:
            tag = resolution or "1080p"
            out = assets / f"torus_{tag}.mp4"
        else:
            out = assets / "torus_construct.gif"

        print(
            f"[animate-torus] mode={args.mode} steps={args.num_steps} "
            f"fps={args.fps} resolution={resolution} encoder={args.encoder} m={modulus}"
        )
        print(
            "  sequence: (1) construct all steps → "
            "(2) short hold → (3) accelerate-rotate 10s"
        )
        anim = animate_torus_projection(
            num_steps=args.num_steps,
            step=step,
            method=args.method,
            modulus=modulus,
            style=args.style,
            frames=args.frames,
            fps=args.fps,
            resolution=resolution,
            encoder=args.encoder,
            cq=args.cq,
            mode=args.mode,
            rotate_seconds=10.0,
            save_path=out,
        )
        print(f"[animate-torus] saved → {out}")
        if args.show:
            plt.show()
        else:
            plt.close(anim._fig)  # type: ignore[attr-defined]

    if args.animate:
        # Keep a reference so the animation is not GC'd before save
        anim = animate_circle_steps(
            num_steps=args.num_steps,
            step=step,
            method=args.method,
            modulus=modulus,
            interval=args.interval,
            style=args.style,
            save_gif=args.save_gif or not args.save_mp4,
            save_mp4=args.save_mp4,
            save_path=Path(args.out) if args.out else None,
            assets_dir=assets,
        )
        print("[animate] done")
        if args.show:
            plt.show()
        else:
            plt.close(anim._fig)  # type: ignore[attr-defined]

    if args.interactive:
        out = Path(args.out) if args.out else assets / "interactive_circle.html"
        plot_interactive_plotly(
            num_steps=args.num_steps,
            step=step,
            method=args.method,
            modulus=modulus,
            save_html=out,
        )
        print(f"[interactive] open in browser → {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
