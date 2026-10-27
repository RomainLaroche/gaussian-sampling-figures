"""Gaussian density capping: exact population moments and retention curves.

Run with: uv run --with numpy --with matplotlib python pollarded_gaussian.py
All outputs are written next to this script. No input data are required.
"""

from pathlib import Path
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, PercentFormatter
import numpy as np


OUT = Path(__file__).resolve().parent
BLUE = "#146f85"
ORANGE = "#dd7734"
GRAY = "#667085"
DARK = "#182230"


def phi(a):
    return math.exp(-a * a / 2) / math.sqrt(2 * math.pi)


def survival(a):
    return 0.5 * math.erfc(a / math.sqrt(2))


def moments(a):
    """Return (retained fraction, STD / original STD) for cap height phi(a)."""
    retained = 2 * (a * phi(a) + survival(a))
    second_moment = retained + 2 * a**3 * phi(a) / 3
    return retained, math.sqrt(second_moment / retained)


def threshold_for_factor(factor):
    """Invert the monotonic population STD curve by bisection."""
    if not 1 <= factor <= 10:
        raise ValueError("Supported STD factors are between 1 and 10.")
    if factor == 1:
        return 0.0
    lo, hi = 0.0, 2 * factor
    for _ in range(90):
        midpoint = (lo + hi) / 2
        if moments(midpoint)[1] < factor:
            lo = midpoint
        else:
            hi = midpoint
    return (lo + hi) / 2


def verify_by_integration():
    """Check the closed forms against direct integration of the capped PDF."""
    z = np.linspace(-12, 12, 600001)
    original = np.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    for factor in [1, 1.25, 1.5, 2, 2.5, 3]:
        a = threshold_for_factor(factor)
        retained, actual_factor = moments(a)
        unnormalized = np.minimum(original, phi(a))
        integrated_retained = np.trapezoid(unnormalized, z)
        mean = np.trapezoid(z * unnormalized, z) / integrated_retained
        variance = np.trapezoid(z**2 * unnormalized, z) / integrated_retained
        assert math.isclose(integrated_retained, retained, rel_tol=1e-8)
        assert abs(mean) < 1e-12
        assert math.isclose(math.sqrt(variance), factor, rel_tol=1e-8)
        assert math.isclose(actual_factor, factor, rel_tol=1e-12)
    print("Direct numerical integration verified retention, mean, and STD.")


def write_csv(factors, filename):
    with (OUT / filename).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "std_factor", "cap_half_width_in_original_std", "cap_height_in_z_units",
            "retained_fraction", "discarded_fraction", "accept_probability_at_mean",
        ])
        for factor in factors:
            a = threshold_for_factor(float(factor))
            retained, _ = moments(a)
            writer.writerow([factor, a, phi(a), retained, 1 - retained, math.exp(-a*a/2)])


def make_figure():
    factor = 2.0
    a = threshold_for_factor(factor)
    retained, _ = moments(a)
    loss = 1 - retained
    cap = phi(a)
    z = np.linspace(-5.5, 5.5, 2501)
    p = np.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    capped = np.minimum(p, cap)
    q = capped / retained
    factors = np.linspace(1, 3, 401)
    retention = np.array([moments(threshold_for_factor(float(k)))[0] for k in factors])

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "text.color": DARK,
        "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.titlesize": 12,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2))
    fig.subplots_adjust(left=0.075, right=0.97, bottom=0.09, top=0.845, wspace=0.27, hspace=0.43)
    fig.suptitle("Capping a Gaussian: twice the STD retains only 2.00% of samples", x=0.075,
                 y=0.967, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.075, 0.923,
             "Randomly thin the dense center to a flat cap; keep every tail sample. "
             "Symmetry preserves the population mean.", fontsize=11, color=GRAY)

    ax = axes[0, 0]
    ax.set_title("A  Remove the peak of the original density", loc="left", pad=12)
    ax.fill_between(z, capped, p, color=ORANGE, alpha=0.30, label=f"Discarded: {loss:.2%}")
    ax.fill_between(z, 0, capped, color=BLUE, alpha=0.8, label=f"Retained: {retained:.2%}")
    ax.plot(z, p, color=GRAY, lw=1.9, label="Original Gaussian")
    ax.plot(z, capped, color=BLUE, lw=1.7)
    ax.set(xlim=(-5.5, 5.5), ylim=(0, 0.43), xlabel=r"Original STD units, $z=(x-\mu)/\sigma$",
           ylabel="Density before renormalizing")
    ax.set_xticks([-5, -3, 0, 3, 5])
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    zoom = ax.inset_axes([0.68, 0.52, 0.29, 0.35])
    zoom.fill_between(z, 0, capped, color=BLUE, alpha=0.25)
    zoom.plot(z, p, color=GRAY, lw=1, ls="--")
    zoom.plot(z, capped, color=BLUE, lw=1.7)
    zoom.axhline(cap, color=ORANGE, lw=1, ls=":")
    zoom.set(xlim=(-5, 5), ylim=(0, cap * 1.45), xticks=[-a, 0, a], yticks=[0, cap])
    zoom.set_xticklabels([f"−{a:.2f}", "0", f"{a:.2f}"], fontsize=7)
    zoom.set_yticklabels(["0", f"{cap:.4f}"], fontsize=7)
    zoom.set_title("Cap: vertical zoom", fontsize=9, fontweight="normal", pad=6)

    ax = axes[0, 1]
    ax.set_title("B  Renormalize the retained samples", loc="left", pad=12)
    ax.plot(z, p, color=GRAY, lw=1.9, ls="--", label=r"Original: STD $=\sigma$")
    ax.fill_between(z, 0, q, color=BLUE, alpha=0.12)
    ax.plot(z, q, color=BLUE, lw=2.4, label=r"Capped + normalized: STD $=2\sigma$")
    for edge in [-a, a]:
        ax.vlines(edge, 0, cap / retained, color=BLUE, alpha=0.5, ls=":", lw=1)
    ax.annotate("", xy=(a, 0.21), xytext=(-a, 0.21),
                arrowprops={"arrowstyle": "<->", "color": BLUE, "lw": 1.3})
    ax.text(0, 0.222, r"Flat within $\mu\,\pm\,3.137\sigma$", ha="center", color=BLUE, fontsize=10)
    ax.set(xlim=(-5.5, 5.5), ylim=(0, 0.43), xlabel=r"Original STD units, $z=(x-\mu)/\sigma$",
           ylabel="Normalized density")
    ax.set_xticks([-5, -3, 0, 3, 5])
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    ax = axes[1, 0]
    ax.set_title("C  Data discarded versus target STD factor", loc="left", pad=12)
    ax.plot(factors, 1 - retention, color=ORANGE, lw=2.6)
    ax.fill_between(factors, 0, 1 - retention, color=ORANGE, alpha=0.10)
    ax.plot([1, 2, 2], [loss, loss, 0], color=ORANGE, ls=":", lw=1.2)
    ax.scatter([2], [loss], color=ORANGE, s=45, zorder=3)
    ax.annotate(f"2× STD → {100 * loss:.2f}% discarded", (2, loss), xytext=(1.50, 0.67),
                fontsize=11, color=ORANGE, fontweight="bold",
                arrowprops={"arrowstyle": "-", "color": ORANGE})
    ax.set(xlim=(1, 3), ylim=(0, 1.04), xlabel=r"STD factor, $k=\sigma_{\rm kept}/\sigma$",
           ylabel="Fraction of original data discarded")
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(alpha=0.16)

    ax = axes[1, 1]
    ax.set_title("D  Data retained (log scale shows the cost)", loc="left", pad=12)
    ax.plot(factors, retention, color=BLUE, lw=2.6)
    ax.set_yscale("log")
    ax.plot([1, 2, 2], [retained, retained, 1e-5], color=BLUE, ls=":", lw=1.2)
    ax.scatter([2], [retained], color=BLUE, s=45, zorder=3)
    ax.annotate(f"2× STD → {100 * retained:.3f}% kept\n≈ 1 sample in {1 / retained:.1f}",
                (2, retained), xytext=(1.10, 0.00035), fontsize=11, color=BLUE,
                arrowprops={"arrowstyle": "-", "color": BLUE})
    ax.set(xlim=(1, 3), ylim=(1e-5, 1.5), xlabel=r"STD factor, $k=\sigma_{\rm kept}/\sigma$",
           ylabel="Fraction of original data retained")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{100*y:g}%"))
    ax.grid(which="major", alpha=0.16)
    for ax in axes[1]:
        ax.set_xticks(np.arange(1, 3.01, 0.25))

    fig.text(0.075, 0.025,
             r"Model: retained density $\min\{\phi(z),\phi(a)\}$. "
             "Population calculations; finite datasets fluctuate. The retained distribution has a flat top.",
             color=GRAY, fontsize=10)
    for extension in ["png", "svg", "pdf"]:
        fig.savefig(OUT / f"gaussian_density_capping.{extension}", dpi=190)
    plt.close(fig)
    write_csv(factors, "gaussian_density_capping_curve.csv")
    selected = [1, 1.1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3]
    write_csv(selected, "gaussian_density_capping_table.csv")
    for k in selected:
        a_k = threshold_for_factor(k)
        r_k, _ = moments(a_k)
        print(f"STD factor {k:g}: keep {100*r_k:.8f}%, discard {100*(1-r_k):.8f}%")
    print(f"For k=2: a={a:.12f}, cap={cap:.12f}, center keep={math.exp(-a*a/2):.12f}")


if __name__ == "__main__":
    verify_by_integration()
    make_figure()
