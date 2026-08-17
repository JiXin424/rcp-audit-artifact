#!/usr/bin/env python3
"""Shared figure style and canonical visual encoding for the LRE paper figures.

Encoding (fixed across all figures):
  REC                  dark-gray square
  TN-PURE              orange-red diamond
  TN-PTCOMP            purple triangle
  released evaluator   orange-red large star
  reconstructions      blue circles
  original reference   orange solid
  human reference      green hollow
  degenerate           light-gray crosses
  zero line            black dashed

All figures export vector PDF; fonts >= 8 pt at final size.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- canonical palette (colorblind-safe + shape-coded) ----
C_REC = "#404040"          # dark gray
C_TNPURE = "#d35400"       # orange-red
C_TNPTCOMP = "#7b2fbe"     # purple
C_RELEASED = "#d35400"     # orange-red (star)
C_RECO = "#2b6cb0"         # blue
C_ORIG = "#d35400"         # orange solid
C_HUMAN = "#2f8f46"        # green hollow
C_DEGEN = "#b0b0b0"        # light gray
C_ZERO = "#000000"         # zero line

M_REC = "s"                # square
M_TNPURE = "D"             # diamond
M_TNPTCOMP = "^"           # triangle
M_RELEASED = "*"           # star
M_RECO = "o"               # circle
M_DEGEN = "x"              # cross

# family markers for Fig 3B
FAM_MARKERS = {
    "faithful": "P",
    "reconstructions": "o",
    "config_faithful": "s",
    "rescue": "^",
    "distillation": "D",
    "released": "*",
}

FAM_COLORS = {
    "faithful": "#d62728",
    "reconstructions": C_RECO,
    "config_faithful": "#2f8f46",
    "rescue": "#8c5a2b",
    "distillation": "#7b2fbe",
    "released": C_RELEASED,
}


def setup(scale=1.0, dpi=200):
    """Apply the paper-wide matplotlib defaults."""
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9 * scale,
        "axes.titlesize": 9 * scale,
        "axes.labelsize": 9 * scale,
        "xtick.labelsize": 8 * scale,
        "ytick.labelsize": 8 * scale,
        "legend.fontsize": 8 * scale,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.edgecolor": "#333333",
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.constrained_layout.use": True,
    })


def panel_label(ax, label, x=-0.22, y=1.06):
    """Bold (a)/(b)/(c) panel label."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left")


def errorbar(ax, x, y, xerr, marker, color, ms=5.5, mew=1.0,
             capsize=2.5, elinewidth=0.9, **kw):
    ax.errorbar(x, y, xerr=xerr, fmt=marker, color=color, ms=ms, mew=mew,
                capsize=capsize, elinewidth=elinewidth, zorder=3, **kw)
