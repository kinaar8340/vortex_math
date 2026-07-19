# Vortex Math · Unit Circle Visualization

Map **positions on the unit circle** obtained by stepping arc lengths of exactly **`9/π` radians** onto **vortex math** concepts: digital roots, the doubling circuit **1-2-4-8-7-5**, and the special role of **3-6-9**.

## Why 9/π?

The unit circle has circumference `2π`. Advancing by a fixed arc `Δθ = 9/π` each step yields the orbit

```text
θ_k = k · (9/π)  (mod 2π)
(x_k, y_k) = (cos θ_k, sin θ_k)
```

The factor **9** echoes the digital-root base of vortex math (digits collapse mod 9 into 1–9). **π** is the circle’s own constant. Relative to a full turn `2π`, this step produces an **irrational rotation** in the usual dense-orbit sense: long sequences fill the circumference densely rather than closing on a short regular polygon (contrast with equal spacing `2π/9`).

Each point is then colored by a **vortex digit** (1–9)—by default the digital root of the step index—so the geometry of the circle and the numerology of the doubling / Trinity patterns can be seen together.

### Vortex concepts used here

| Concept | Meaning in this project |
|--------|-------------------------|
| **Digital root** | Iterated digit sum → single digit 1–9 (`n mod 9`, with multiples of 9 → 9) |
| **Doubling circuit** | `1 → 2 → 4 → 8 → 7 → 5 → 1` (digital roots of powers of 2) |
| **3-6-9 (Trinity)** | Digits outside pure doubling; often drawn as a control / axis (9 highlighted at center) |

## Project layout

```text
vortex_math/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── core.py          # Pure math (digital root, sequences, circle positions)
│   ├── visualize.py     # Matplotlib + Plotly plots & animations
│   └── main.py          # CLI entry point
├── notebooks/
│   └── exploration.ipynb
├── assets/              # Generated PNGs, GIFs, HTML
└── tests/
    └── test_core.py
```

## Install

```bash
cd ~/Projects/vortex_math
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional for tests:

```bash
pip install pytest
```

## Run

From the project root:

```bash
# Default demo — writes key figures under assets/
python src/main.py --demo

# Stepped unit circle (colored by vortex digit)
python src/main.py --plot-steps --num-steps 100

# Side-by-side vortex star + orbit
python src/main.py --vortex-flow --num-steps 80

# Density / fill of a long orbit
python src/main.py --density --num-steps 2000

# 3D toroidal projection (black high-contrast wireframe)
python src/main.py --torus --num-steps 300

# Torus construction (default): watch the 9/π winding build point-by-point
# Default resolution is 1080p MP4
python src/main.py --animate-torus --steps 300

# GIF preview of the same construction
python src/main.py --animate-torus --steps 200 --save-gif

# Optional: spin the finished orbit instead of constructing it
python src/main.py --animate-torus --mode spin --steps 300

# Higher res (NVENC on RTX 4090)
python src/main.py --animate-torus --4k --steps 400 --fps 30
# encoder: --encoder auto|h264_nvenc|hevc_nvenc|av1_nvenc|libx264

# Animation → assets/circle_steps.gif
python src/main.py --animate --steps 300 --save-gif

# Interactive Plotly HTML
python src/main.py --interactive

# Custom step / mapping
python src/main.py --plot-steps --step 9/pi --method angle_bin --num-steps 150
```

### Python API

```python
from src.core import (
    digital_root,
    vortex_doubling_sequence,
    circle_positions,
    position_to_vortex_digit,
    DEFAULT_STEP_RADIANS,
)
from src.visualize import plot_unit_circle_with_steps, animate_circle_steps

print(digital_root(247))                 # 4
print(vortex_doubling_sequence(6))       # [1, 2, 4, 8, 7, 5]
x, y = circle_positions(50)              # step default = 9/π

fig = plot_unit_circle_with_steps(num_steps=100, save_path="assets/out.png")
```

## Current visualizations

| Output | Description |
|--------|-------------|
| `assets/unit_circle_steps.png` | Scatter on the unit circle, colored by vortex digit; labels; 1–9 overlay + doubling star |
| `assets/vortex_flow.png` | Classic vortex star beside the 9/π orbit |
| `assets/density_heatmap.png` | Plane density + angular histogram (dense fill) |
| `assets/torus_projection.png` | Parametric 3D torus winding (black wireframe, vortex-colored points) |
| `assets/torus_1080p.mp4` | **Default:** point-by-point torus construction at 1080p (`--animate-torus`) |
| `assets/torus_construct.gif` | Same construction as GIF (`--animate-torus --save-gif`) |
| `assets/torus_4k.mp4` | 4K construction (or spin with `--mode spin --4k`) |
| `assets/interactive_circle.html` | Plotly figure with frames / slider over `num_steps` |
| `assets/circle_steps.gif` | Progressive animation of the orbit (via `--animate --save-gif`) |

### Mapping methods (`--method`)

- `step_index` (default) — digital root of the step index (step 0 → 9)
- `angle_bin` — nine equal arcs: `floor((θ/2π)·9)+1`
- `sin_dr` / `cos_dr` — digital root of scaled `|sin θ|` / `|cos θ|`
- `doubling_cycle` — cycle 1-2-4-8-7-5 by step index

Register custom mappings with `core.register_mapping(name, fn)`.

## Tests

```bash
pytest tests/ -v
```

## High-resolution / GPU video (RTX 4090)

Matplotlib still **draws** frames on the CPU, but **encoding** can use your NVIDIA GPU via ffmpeg NVENC — very fast on an RTX 4090 for 4K/8K MP4.

Requirements: system `ffmpeg` built with NVENC (you already have `h264_nvenc` / `hevc_nvenc` / `av1_nvenc` if `ffmpeg -encoders | grep nvenc` lists them).

```python
from src.visualize import animate_torus_projection, nvenc_available

print("NVENC:", nvenc_available())  # True on this machine

animate_torus_projection(
    num_steps=300,
    mode="construct",      # default: build the winding step-by-step
    resolution="1080p",    # default
    fps=30,
    encoder="auto",        # prefers h264_nvenc → hevc_nvenc → libx264
    cq=18,
    save_path="assets/torus_1080p.mp4",
)
```

| Approach | GPU use | Quality | Notes |
|----------|---------|---------|--------|
| Matplotlib + NVENC | Encoding | Very good | Default path in this repo |
| PyVista / VTK | Full render | Excellent | Optional next step |
| Manim / Blender | Full | Cinematic | Future |

**8K note:** `h264_nvenc` is limited to ~4096×4096 on GeForce cards, so **8K (7680×4320) cannot use H.264 NVENC**. The pipeline auto-upgrades to `hevc_nvenc` (or `av1_nvenc` / `libx265`) when the frame size exceeds that limit. Prefer:

```bash
python src/main.py --animate-torus --8k --encoder hevc_nvenc --cq 16
# or simply --8k --encoder auto  (picks hevc_nvenc for 8K)
```

## Design notes

- **Math vs. rendering**: all pure functions live in `src/core.py`; plotting only in `src/visualize.py`.
- **Configurable**: step size, mapping method, and style (`dark` / `light`) are parameters (see also `default_config()`).
- **Irrational rotation**: large `num_steps` intentionally produces dense covering; density plots make that visible.
- **Extensible**: add mappings, new projections, or sound without rewriting the orbit math.

## Ideas for next iterations

Once the base visualizations are working, iterate with prompts like:

- “Add a 3D toroidal projection of the stepped points.” *(starter already in `plot_torus_projection`)*
- “Color points by digital root of sin(θ) or other trig functions.” *(see `sin_dr` / `cos_dr`)*
- “Create an interactive dashboard with sliders using Dash or Streamlit.”
- “Map the sequence to musical notes or frequencies and generate audio.”
- “Compare multiple step sizes (including `2π/9` for equal spacing vs `9/π`).”
- “Rodin-coil inspired multi-winding on a torus with 3-6-9 highlighted as a poloidal axis.”
- “Export SVG layers for print / laser (number circle + orbit separately).”

## License / context

Educational visualization project. Vortex math here is treated as a **numerological / geometric motif** (digital roots and doubling patterns), not as a claim about physics. Explore, plot, and extend freely.
