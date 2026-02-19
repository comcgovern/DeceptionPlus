"""
preview_social_image.py
-----------------------
Generates a preview PNG matching the redesigned create_social_media_graphics()
output — 1200x675 px (16:9), suitable for Twitter / Bluesky.

Run:  python3 preview_social_image.py
Output: output/test_visuals/preview_*.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
import datetime

os.makedirs("output/test_visuals", exist_ok=True)

today = datetime.date.today().strftime("%B %d, %Y")
today_short = datetime.date.today().strftime("%Y-%m-%d")

# ---------------------------------------------------------------------------
# Mock data (mirrors generate_test_image.R)
# ---------------------------------------------------------------------------
starters_unpred = [
    ("Gerrit Cole",      128),
    ("Spencer Strider",  121),
    ("Zack Wheeler",     116),
    ("Pablo Lopez",      112),
    ("Logan Webb",       109),
]

starters_pred = [
    ("Nestor Cortes",    81),
    ("Chris Sale",       85),
    ("Joe Musgrove",     88),
    ("Sandy Alcantara",  91),
    ("Luis Castillo",    94),
]

relievers_unpred = [
    ("Edwin Diaz",       133),
    ("Emmanuel Clase",   125),
    ("Ryan Helsley",     119),
    ("Josh Hader",       114),
    ("Devin Williams",   110),
]

relievers_pred = [
    ("Clay Holmes",      78),
    ("Pete Fairbanks",   84),
    ("Bryan Abreu",      89),
    ("Alexis Diaz",      93),
    ("David Bednar",     97),
]

CHARTS = [
    dict(
        data=starters_unpred,
        title="Least Predictable Starters",
        subtitle=f"{today}  ·  min 65 pitches",
        caption=f"Higher = less predictable  ·  100 = league average  ·  @PredictPlus  ·  data: Baseball Savant",
        color_low="#4361ee", color_high="#7209b7",
        fname=f"preview_starters_top5_unpredictable_{today_short}.png",
        desc_order=True,
    ),
    dict(
        data=starters_pred,
        title="Most Predictable Starters",
        subtitle=f"{today}  ·  min 65 pitches",
        caption=f"Lower = more predictable  ·  100 = league average  ·  @PredictPlus  ·  data: Baseball Savant",
        color_low="#e63946", color_high="#f4a261",
        fname=f"preview_starters_top5_predictable_{today_short}.png",
        desc_order=False,
    ),
    dict(
        data=relievers_unpred,
        title="Least Predictable Relievers",
        subtitle=f"{today}  ·  min 15 pitches",
        caption=f"Higher = less predictable  ·  100 = league average  ·  @PredictPlus  ·  data: Baseball Savant",
        color_low="#2a9d8f", color_high="#264653",
        fname=f"preview_relievers_top5_unpredictable_{today_short}.png",
        desc_order=True,
    ),
    dict(
        data=relievers_pred,
        title="Most Predictable Relievers",
        subtitle=f"{today}  ·  min 15 pitches",
        caption=f"Lower = more predictable  ·  100 = league average  ·  @PredictPlus  ·  data: Baseball Savant",
        color_low="#e76f51", color_high="#f4a261",
        fname=f"preview_relievers_top5_predictable_{today_short}.png",
        desc_order=False,
    ),
]


def make_gradient_colors(values, hex_low, hex_high):
    """Map a list of values to gradient colours between hex_low and hex_high."""
    lo = np.array(mcolors.to_rgb(hex_low))
    hi = np.array(mcolors.to_rgb(hex_high))
    vmin, vmax = min(values), max(values)
    if vmin == vmax:
        return [hi for _ in values]
    t_vals = [(v - vmin) / (vmax - vmin) for v in values]
    return [lo + t * (hi - lo) for t in t_vals]


def draw_chart(cfg):
    # --- dimensions: 1200x675 px ---
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=100)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")

    data = cfg["data"]
    # Sort so highest bar is at top
    if cfg["desc_order"]:
        ordered = sorted(data, key=lambda x: x[1])          # ascending → top bar = highest
    else:
        ordered = sorted(data, key=lambda x: -x[1])         # descending → top bar = lowest

    names  = [f"#{i+1}  {n}" for i, (n, _) in enumerate(reversed(ordered))]
    scores = [s for (_, s) in reversed(ordered)]
    y_pos  = np.arange(len(names))

    x_track = max(scores + [110]) * 1.20
    x_left  = min(scores + [90])  * 0.94

    colors = make_gradient_colors(scores, cfg["color_low"], cfg["color_high"])

    # Track bars (light grey)
    ax.barh(y_pos, [x_track] * len(scores), left=x_left,
            height=0.55, color="#efefef", zorder=1)

    # Data bars
    ax.barh(y_pos, [s - x_left for s in scores], left=x_left,
            height=0.55, color=colors, zorder=2)

    # Score labels
    for y, s in zip(y_pos, scores):
        ax.text(s + (x_track - x_left) * 0.022, y, f"{s:.0f}",
                va="center", ha="left",
                fontsize=11, fontweight="bold", color="#333333", zorder=3)

    # League average dashed line + AVG label
    ax.axvline(100, color="#bbbbbb", linewidth=1.0, linestyle="--", zorder=4)
    ax.text(100, -0.65, "AVG", ha="center", va="top",
            fontsize=8, color="#bbbbbb")

    # Y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=12, fontweight="bold", color="#2d2d2d")

    # X-axis
    ax.set_xlim(x_left, x_track)
    ax.set_xlabel("Predict+  (100 = league average)", fontsize=10, color="#888888",
                  labelpad=6)
    ax.tick_params(axis="x", labelsize=10, colors="#999999")
    ax.tick_params(axis="y", length=0)

    # Grid (vertical only, behind bars)
    ax.xaxis.grid(True, color="#eeeeee", linewidth=0.6, zorder=0)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)

    # Spines — keep only bottom
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#dddddd")

    # Title & subtitle (left-aligned)
    fig.text(0.07, 0.95, cfg["title"],
             fontsize=20, fontweight="bold", color="#1a1a2e",
             ha="left", va="top", transform=fig.transFigure)
    fig.text(0.07, 0.875, cfg["subtitle"],
             fontsize=12, color="#666666",
             ha="left", va="top", transform=fig.transFigure)

    # Caption (bottom-right)
    fig.text(0.97, 0.02, cfg["caption"],
             fontsize=8.5, color="#aaaaaa",
             ha="right", va="bottom", transform=fig.transFigure)

    # Tight layout accounting for supra-axes title
    plt.subplots_adjust(left=0.22, right=0.96, top=0.78, bottom=0.13)

    out_path = os.path.join("output/test_visuals", cfg["fname"])
    fig.savefig(out_path, dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out_path}")


print(f"\nGenerating 4 preview images (1200x675 px, 16:9) ...\n")
for chart_cfg in CHARTS:
    draw_chart(chart_cfg)

print("\nDone. Open output/test_visuals/ to review.\n")
