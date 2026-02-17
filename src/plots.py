"""Visualization functions.

Each plot function is looked up by name from the config's plots list.
All functions take the same arguments: results dict, window label,
settings dict, and output directory.

Adapted from trading/scripts/best_three_test.py plot functions.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# Minimum trading days in a year to include in annual returns chart
MIN_DAYS_FOR_ANNUAL = 20

# Default colors — assigned cyclically to strategies/benchmarks
DEFAULT_COLORS = [
    "#2E7D32", "#66BB6A", "#4A148C", "#CE93D8",
    "#FF5722", "#FF9800", "#616161", "#1565C0",
    "#00897B", "#EF5350",
]


def equity_curves(results: dict, window: str, settings: dict,
                  output_dir: str, **kwargs) -> str:
    """Plot equity curves and drawdowns for all portfolios in a window.

    Args:
        results: Full results dict from compare.run_comparison().
        window: Window label (key into results).
        settings: Config settings dict.
        output_dir: Directory to save the plot.

    Returns:
        Path to saved PNG.
    """
    window_data = results.get(window, {})
    if not window_data:
        return None

    names = list(window_data.keys())
    colors = _assign_colors(names)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[2, 1])

    # Equity curves
    ax = axes[0]
    for name in names:
        eq = window_data[name]["equity_curve"]
        ax.plot(eq.index, eq.values, label=name, color=colors[name],
                linewidth=2.0, alpha=0.9)

    ax.set_yscale("log")
    ax.set_ylim(bottom=max(1.0, ax.get_ylim()[0]))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    ax.set_title(f"Equity Curves — {window}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")

    # Drawdowns
    ax = axes[1]
    for name in names:
        eq = window_data[name]["equity_curve"]
        dd = eq / eq.cummax() - 1
        ax.plot(dd.index, dd.values * 100, label=name,
                color=colors[name], linewidth=1.5, alpha=0.9)

    ax.set_title("Drawdowns", fontsize=12, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=8, loc="lower left", ncol=3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{_slugify(window)}_equity_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def drawdowns(results: dict, window: str, settings: dict,
              output_dir: str, **kwargs) -> str:
    """Standalone drawdown chart for all portfolios in a window."""
    window_data = results.get(window, {})
    if not window_data:
        return None

    names = list(window_data.keys())
    colors = _assign_colors(names)

    fig, ax = plt.subplots(figsize=(16, 6))
    for name in names:
        eq = window_data[name]["equity_curve"]
        dd = eq / eq.cummax() - 1
        ax.plot(dd.index, dd.values * 100, label=name,
                color=colors[name], linewidth=1.5, alpha=0.9)

    ax.set_title(f"Drawdowns — {window}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=9, loc="lower left", ncol=3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{_slugify(window)}_drawdowns.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def annual_returns(results: dict, window: str, settings: dict,
                   output_dir: str, **kwargs) -> str:
    """Grouped bar chart of annual returns for all portfolios."""
    window_data = results.get(window, {})
    if not window_data:
        return None

    names = list(window_data.keys())
    colors = _assign_colors(names)

    # Collect all years
    all_years = set()
    for name in names:
        ret = window_data[name]["returns"]
        all_years.update(ret.index.year.unique())
    years = sorted(all_years)

    x = np.arange(len(years))
    n_bars = len(names)
    width = 0.80 / max(n_bars, 1)

    fig, ax = plt.subplots(figsize=(18, 6))
    for i, name in enumerate(names):
        ret = window_data[name]["returns"]
        annual = []
        for yr in years:
            yr_ret = ret[ret.index.year == yr]
            if len(yr_ret) > MIN_DAYS_FOR_ANNUAL:
                annual.append(float((1 + yr_ret).prod() - 1) * 100)
            else:
                annual.append(0)
        ax.bar(x + i * width, annual, width, label=name,
               color=colors[name], alpha=0.85)

    ax.set_title(f"Annual Returns — {window}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Return (%)")
    ax.set_xticks(x + width * (n_bars - 1) / 2)
    ax.set_xticklabels(years, rotation=45)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{_slugify(window)}_annual_returns.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def risk_return_scatter(results: dict, window: str, settings: dict,
                        output_dir: str, **kwargs) -> str:
    """Scatter plot: x=MaxDD, y=CAGR, one dot per portfolio."""
    window_data = results.get(window, {})
    if not window_data:
        return None

    names = list(window_data.keys())
    colors = _assign_colors(names)

    fig, ax = plt.subplots(figsize=(10, 8))
    for name in names:
        m = window_data[name]["metrics"]
        dd = m.get("max_drawdown", 0) * 100
        ret = m.get("cagr", 0) * 100
        ax.scatter(dd, ret, s=120, color=colors[name], zorder=3)
        ax.annotate(name, (dd, ret), fontsize=9,
                    xytext=(5, 5), textcoords="offset points")

    ax.set_title(f"Risk vs Return — {window}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Max Drawdown (%)")
    ax.set_ylabel("CAGR (%)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{_slugify(window)}_risk_return.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def rolling_12m(results: dict, window: str, settings: dict,
                output_dir: str, **kwargs) -> str:
    """Rolling 12-month Sharpe ratio for all portfolios."""
    window_data = results.get(window, {})
    if not window_data:
        return None

    names = list(window_data.keys())
    colors = _assign_colors(names)
    roll_window = 252

    fig, ax = plt.subplots(figsize=(16, 6))
    for name in names:
        ret = window_data[name]["returns"]
        if len(ret) < roll_window:
            continue
        min_periods = int(roll_window * 0.8)
        rs = (ret.rolling(roll_window, min_periods=min_periods).mean() * 252) / (
            ret.rolling(roll_window, min_periods=min_periods).std() * np.sqrt(252)
        )
        ax.plot(rs.index, rs.values, label=name, color=colors[name],
                linewidth=1.8, alpha=0.9)

    ax.set_title(f"Rolling 1-Year Sharpe — {window}",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axhline(y=1, color="green", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{_slugify(window)}_rolling_12m.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ===================================================================
# HELPERS
# ===================================================================

def _assign_colors(names: list) -> dict:
    """Assign colors to portfolio names."""
    return {name: DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            for i, name in enumerate(names)}


def _slugify(text: str) -> str:
    """Convert window label to filename-safe slug."""
    return text.lower().replace(" ", "_").replace("-", "_")
