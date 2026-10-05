"""Restyle manuscript Figure 2 without reconstructing numerical data.

The complete trajectory corresponding to ntk.png is not available. The old
notebook's printed values are from a different run and must not be substituted.
This script preserves the colored curve pixels in the manuscript's original
image, replacing only axes, ticks, grid, and legend with editable vector artwork.
The source PNG and historical notebooks remain unchanged. No digitization,
interpolation, numerical evaluation, or training is performed.
"""

from pathlib import Path
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import BboxImage
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, NullLocator
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "ntk.png"
SOURCE_SHA256 = "1e2646fa1ebd137e83be02b7b071d764b1a329f3581fffa98334fb32c3fff4d9"


def original_curve_pixels():
    """Isolate the two original colored traces, excluding the old legend.

    Original spine centers: x=83,580; y=5,302 (top-down image coordinates).
    Include a one-pixel transparent border at those centers so that BboxImage
    maps the source plot coordinates onto the new axes without changing scale.
    Chroma selects colored strokes rather than the grayscale grid/background.
    RGB values and pixel positions of retained strokes are copied unchanged.
    """
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Source image changed; recheck the plot-frame geometry.")
    rgb = np.asarray(Image.open(SOURCE).convert("RGB"))
    if rgb.shape != (348, 590, 3):
        raise ValueError(f"Unexpected source image dimensions: {rgb.shape}")
    red, green, blue = np.moveaxis(rgb.astype(int), -1, 0)
    rows, columns = np.indices(rgb.shape[:2])
    in_curve_columns = (columns >= 100) & (columns <= 561)
    interior = (in_curve_columns & (rows >= 20) & (rows <= 65)
                & (blue - red > 24) & (blue - green > 12))
    boundary = (in_curve_columns & (rows >= 274) & (rows <= 284)
                & (red - blue > 35) & (green - blue > 25))
    if interior.sum() < 900 or boundary.sum() < 900:
        raise ValueError("Too few curve pixels; inspect the source image.")
    # Pixel-center bounds coincide exactly with the original spine centers.
    # BboxImage uses pixel-edge bounds, hence the half-pixel extent adjustment
    # when fitting its rectangle to the axes in main().
    rgba = np.zeros((298, 498, 4), dtype=np.uint8)
    rgba[..., :3] = rgb[5:303, 83:581]
    rgba[..., 3] = ((interior | boundary)[5:303, 83:581] * 255)
    assert np.array_equal(rgba[..., :3], rgb[5:303, 83:581])
    return rgba


def main():
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.labelsize": 13, "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5, "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    }):
        fig, ax = plt.subplots(figsize=(6.0, 3.55))
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.19, top=0.965)
        # The scale and data extent are those of the original notebook plot.
        ax.set_xlim(-4.95, 103.95)
        ax.set_yscale("log")
        ax.set_ylim(1e-13, 2.0)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.yaxis.set_major_locator(FixedLocator([1, 1e-3, 1e-6, 1e-9, 1e-12]))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("Outer iteration", labelpad=6)
        ax.set_ylabel(r"Relative change $\rho_Q(k)$", labelpad=7)
        ax.tick_params(direction="out", length=3.5, width=0.8, pad=4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#555555")
        ax.set_axisbelow(True)
        ax.grid(which="major", color="#dce2e8", linewidth=0.6,
                linestyle=(0, (3, 3)))

        # Preserve original curve pixels; do not infer a numerical trajectory.
        from matplotlib.transforms import Bbox, TransformedBbox
        bounds = Bbox.from_extents(-0.5 / 497, -0.5 / 297,
                                  1 + 0.5 / 497, 1 + 0.5 / 297)
        trace = BboxImage(TransformedBbox(bounds, ax.transAxes),
                          interpolation="bilinear", origin="upper", zorder=3)
        trace.set_data(original_curve_pixels())
        trace.set_clip_path(ax.patch)
        ax.add_artist(trace)
        ax.legend(handles=[
            Line2D([], [], color="#0072B2", lw=2,
                   label="Interior residual NTK"),
            Line2D([], [], color="#E69F00", lw=2,
                   label="Boundary residual NTK"),
        ], loc="center right", bbox_to_anchor=(0.985, 0.48),
           frameon=True, facecolor="white", edgecolor="none", framealpha=0.97,
           fontsize=12, handlelength=2.1, labelspacing=0.8,
           borderpad=0.6, borderaxespad=0.3)
        destination = HERE / "figure2_burgers_ntk"
        fig.savefig(destination.with_suffix(".pdf"),
                    metadata={"Title": "Relative changes of residual NTK matrices",
                              "Subject": "Original curve pixels with restyled vector labels"})
        fig.savefig(destination.with_suffix(".png"), dpi=300)
        plt.close(fig)
        print("Preserved original curve pixels; saved", destination.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
