"""
CCDF degree distribution plots (log-log and linear) for directed GraphML networks.
"""
import sys
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# settings
FIT_POWER_LAW = True  # set False to skip power law fitting

networks = {
    "C. elegans SCC": {
        "path": "/Users/minghanfan/Desktop/math479/Capstone/celegansgraph/elegans_scc.graphml",
        "weighted": False,
    },
    "Fruit Fly SCC": {
        "path": "/Users/minghanfan/Desktop/math479/Capstone/fruitflygraph/region_FB_giant_scc.graphml",
        "weighted": True,
    },
}

def ccdf(degrees):
    """Return sorted unique values and their CCDF (P(X >= x))."""
    vals = np.sort(degrees)
    n = len(vals)
    unique, counts = np.unique(vals, return_counts=True)
    cum = np.cumsum(counts)
    ccdf_vals = 1.0 - (cum - counts) / n
    return unique, ccdf_vals


def plot_ccdf(ax, degrees, label, color, fit_pl=False):
    """Plot CCDF on log-log axes, optionally with power law fit."""
    degrees = np.array(degrees, dtype=float)
    degrees = degrees[degrees > 0]

    x, y = ccdf(degrees)
    ax.scatter(x, y, s=15, alpha=0.7, color=color, label=label, zorder=3)

    if fit_pl and FIT_POWER_LAW:
        fit = powerlaw.Fit(degrees, discrete=True, verbose=False)
        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
        ax.set_title(f"α={alpha:.2f}, xmin={xmin:.0f}", fontsize=9)


# main
for name, cfg in networks.items():
    G = nx.read_graphml(cfg["path"])

    deg_types = {}
    deg_types["In-degree"] = [d for _, d in G.in_degree()]
    deg_types["Out-degree"] = [d for _, d in G.out_degree()]

    if cfg["weighted"]:
        deg_types["Weighted in-degree"] = [d for _, d in G.in_degree(weight="weight")]
        deg_types["Weighted out-degree"] = [d for _, d in G.out_degree(weight="weight")]

    colors = ["#2176AE", "#D7263D", "#F46036", "#1B998B"]

    for (dtype, degs), color in zip(deg_types.items(), colors):
        suffix = dtype.lower().replace(" ", "_").replace("-", "")
        base = "/Users/minghanfan/Desktop/math479/Capstone/ccdf/ccdf_" + name.lower().replace(" ", "_").replace(".", "") + "_" + suffix

        # log-log
        fig, ax = plt.subplots(figsize=(5, 4.5))
        plot_ccdf(ax, degs, dtype, color, fit_pl=True)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(dtype)
        ax.set_ylabel("CCDF  P(X ≥ x)")
        ax.legend(fontsize=8)
        fig.suptitle(f"{name}: {dtype} CCDF (log-log)", fontsize=13)
        fig.tight_layout()
        fig.savefig(base + "_loglog.png", dpi=200, bbox_inches="tight")
        print(f"Saved: {base}_loglog.png")

        # linear
        fig, ax = plt.subplots(figsize=(5, 4.5))
        plot_ccdf(ax, degs, dtype, color, fit_pl=False)
        ax.set_xlabel(dtype)
        ax.set_ylabel("CCDF  P(X ≥ x)")
        ax.legend(fontsize=8)
        fig.suptitle(f"{name}: {dtype} CCDF (linear)", fontsize=13)
        fig.tight_layout()
        fig.savefig(base + "_linear.png", dpi=200, bbox_inches="tight")
        print(f"Saved: {base}_linear.png")

plt.show()