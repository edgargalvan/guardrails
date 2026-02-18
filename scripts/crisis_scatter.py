#!/usr/bin/env python3
"""Crisis scatter plot: redistribution dial vs benchmarks during GFC and 2022.

Generates one figure with two panels (GFC, 2022), each showing both EW and
Momentum redistribution lines alongside standard benchmarks during the two
sustained multi-asset drawdowns.

Usage:
    python scripts/crisis_scatter.py
"""

import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src import backtest, data, filters as filters_module, metrics
from src import weights as weights_module

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────────────────

CRISIS_WINDOWS = {
    "GFC (2008-2009)": ("2008-01-01", "2009-12-31"),
    "2022 Rate Hike": ("2022-01-01", "2022-12-31"),
}

BENCHMARKS = {
    "SPY": {"SPY": 1.0},
    "60/40": {"SPY": 0.60, "AGG": 0.40},
    "All-Weather-Lite": {"SPY": 0.30, "TLT": 0.40, "GLD": 0.15, "IEF": 0.15},
}

REDISTRIBUTION_LEVELS = [0.0, 0.25, 0.50, 0.75, 1.0]

UNIVERSE = ["SPY", "TLT", "GLD"]


# ── Helpers ─────────────────────────────────────────────────────────────────

def total_return(equity: pd.Series) -> float:
    """Total return as a fraction (e.g., -0.35 for -35%)."""
    if len(equity) < 2:
        return 0.0
    return equity.iloc[-1] / equity.iloc[0] - 1.0


def crisis_max_dd(equity: pd.Series) -> float:
    """Max drawdown over a windowed equity curve (negative fraction)."""
    return metrics.max_drawdown(equity)


def run_strategy_crisis(
    prices: pd.DataFrame,
    cash_prices: pd.Series,
    weight_type: str,
    redistribution_pct: float,
    crisis_start: str,
    crisis_end: str,
    weight_kwargs: dict = None,
    initial_capital: float = 200_000,
    slippage_bps: float = 2.0,
):
    """Run strategy from full data start, trim equity to crisis window."""
    if weight_kwargs is None:
        weight_kwargs = {}

    strat_prices = prices[UNIVERSE].dropna()
    in_mask = filters_module.sma(strat_prices, window=200, frequency="monthly")
    weight_fn = getattr(weights_module, weight_type)
    target_weights = weight_fn(strat_prices, in_mask, **weight_kwargs)

    equity = backtest.run_backtest(
        prices=strat_prices,
        weights=target_weights,
        redistribution_pct=redistribution_pct,
        cash_prices=cash_prices,
        initial_capital=initial_capital,
        slippage_bps=slippage_bps,
    )

    # Trim to crisis window
    start_ts = pd.Timestamp(crisis_start)
    end_ts = pd.Timestamp(crisis_end)
    eq_w = equity[(equity.index >= start_ts) & (equity.index <= end_ts)]

    if len(eq_w) < 2:
        return 0.0, 0.0

    return total_return(eq_w), crisis_max_dd(eq_w)


def run_benchmark_crisis(
    prices: pd.DataFrame,
    allocation: dict,
    crisis_start: str,
    crisis_end: str,
    initial_capital: float = 200_000,
    slippage_bps: float = 2.0,
):
    """Run benchmark from full data start, trim equity to crisis window."""
    equity = backtest.run_benchmark(
        prices=prices,
        allocation=allocation,
        initial_capital=initial_capital,
        rebalance="quarterly",
        slippage_bps=slippage_bps,
    )

    start_ts = pd.Timestamp(crisis_start)
    end_ts = pd.Timestamp(crisis_end)
    eq_w = equity[(equity.index >= start_ts) & (equity.index <= end_ts)]

    if len(eq_w) < 2:
        return 0.0, 0.0

    return total_return(eq_w), crisis_max_dd(eq_w)


# ── Plotting ────────────────────────────────────────────────────────────────

EW_COLOR = "#1565C0"       # blue
MOM_COLOR = "#D32F2F"      # red

BENCHMARK_MARKERS = {
    "SPY": {"marker": "D", "color": "#888888"},
    "60/40": {"marker": "s", "color": "#2E7D32"},
    "All-Weather-Lite": {"marker": "^", "color": "#FF8F00"},
}


def _plot_strategy_line(ax, points, color, label, all_x, all_y):
    """Plot one strategy's redistribution line on an axis."""
    x = [p[2] * 100 for p in points]
    y = [p[1] * 100 for p in points]
    all_x.extend(x)
    all_y.extend(y)

    ax.plot(x, y, color="black", linewidth=0.8, alpha=0.5, zorder=1)
    ax.scatter(x, y, c=color, s=120, edgecolors="black", linewidths=0.5,
               alpha=0.9, zorder=3, label=label)

    for pct, ret, dd in points:
        ax.annotate(f"{pct*100:.0f}%",
                    (dd * 100, ret * 100),
                    fontsize=8, fontweight="bold", color=color,
                    xytext=(7, -4), textcoords="offset points", zorder=4)


def plot_crisis_scatter(
    ew_data: dict,
    mom_data: dict,
    benchmark_data: dict,
    output_path: str,
):
    """Create two-panel crisis scatter with both EW and Momentum lines.

    Parameters
    ----------
    ew_data : dict
        {crisis_label: [(pct, total_ret, max_dd), ...]}
    mom_data : dict
        {crisis_label: [(pct, total_ret, max_dd), ...]}
    benchmark_data : dict
        {crisis_label: {bench_name: (total_ret, max_dd)}}
    output_path : str
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    crisis_labels = list(CRISIS_WINDOWS.keys())

    all_x, all_y = [], []

    for ax, crisis_label in zip(axes, crisis_labels):
        # EW redistribution line
        _plot_strategy_line(ax, ew_data[crisis_label], EW_COLOR,
                            "EW dial", all_x, all_y)

        # Momentum redistribution line
        _plot_strategy_line(ax, mom_data[crisis_label], MOM_COLOR,
                            "Momentum dial", all_x, all_y)

        # Benchmark dots
        bench_points = benchmark_data[crisis_label]
        for bench_name, (ret, dd) in bench_points.items():
            bm = BENCHMARK_MARKERS[bench_name]
            all_x.append(dd * 100)
            all_y.append(ret * 100)
            ax.scatter(dd * 100, ret * 100, marker=bm["marker"],
                       c=bm["color"], s=160, edgecolors="black",
                       linewidths=0.5, alpha=0.9, zorder=3,
                       label=bench_name)
            ax.annotate(bench_name,
                        (dd * 100, ret * 100),
                        fontsize=8, fontweight="bold", color=bm["color"],
                        xytext=(7, 5), textcoords="offset points", zorder=4)

        ax.set_title(crisis_label, fontsize=13, fontweight="bold")
        ax.set_xlabel("Max Drawdown (%)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

    axes[0].set_ylabel("Total Return (%)", fontsize=11)

    # Common axis ranges
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_pad = (x_max - x_min) * 0.12
    y_pad = (y_max - y_min) * 0.12
    for ax in axes:
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # Unified legend from left panel
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Crisis Period Performance: Redistribution Dial vs Benchmarks",
                 fontsize=15, fontweight="bold", y=1.0)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    config_path = os.path.join(PROJECT_ROOT, "configs", "default.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("Loading price data...")
    prices = data.load_prices(config)

    settings = config.get("settings", {})
    cash_vehicle = settings.get("cash_vehicle")
    cash_prices = prices[cash_vehicle] if cash_vehicle and cash_vehicle in prices.columns else None
    initial_capital = settings.get("initial_capital", 200_000)
    slippage_bps = settings.get("costs", {}).get("slippage_bps", 2.0)

    output_dir = os.path.join(PROJECT_ROOT, settings.get("output_dir", "results"))
    os.makedirs(output_dir, exist_ok=True)

    momentum_kwargs = {
        "lookback_days": 252,
        "skip_days": 21,
        "split": [0.70, 0.20, 0.10],
    }

    # ── Benchmarks ──────────────────────────────────────────────────────
    logger.info("\nComputing benchmark results...")
    benchmark_data = {}
    for crisis_label, (c_start, c_end) in CRISIS_WINDOWS.items():
        benchmark_data[crisis_label] = {}
        for bench_name, allocation in BENCHMARKS.items():
            ret, dd = run_benchmark_crisis(
                prices, allocation, c_start, c_end,
                initial_capital=initial_capital,
                slippage_bps=slippage_bps,
            )
            benchmark_data[crisis_label][bench_name] = (ret, dd)
            logger.info("  %s / %s: Return=%.1f%%  MaxDD=%.1f%%",
                        crisis_label, bench_name, ret * 100, dd * 100)

    # ── EW strategy ─────────────────────────────────────────────────────
    logger.info("\nComputing EW strategy across crisis windows...")
    ew_data = {}
    for crisis_label, (c_start, c_end) in CRISIS_WINDOWS.items():
        points = []
        for pct in REDISTRIBUTION_LEVELS:
            ret, dd = run_strategy_crisis(
                prices, cash_prices,
                weight_type="equal",
                redistribution_pct=pct,
                crisis_start=c_start,
                crisis_end=c_end,
                initial_capital=initial_capital,
                slippage_bps=slippage_bps,
            )
            points.append((pct, ret, dd))
            logger.info("  %s / EW pct=%.0f%%: Return=%.1f%%  MaxDD=%.1f%%",
                        crisis_label, pct * 100, ret * 100, dd * 100)
        ew_data[crisis_label] = points

    # ── Momentum strategy ───────────────────────────────────────────────
    logger.info("\nComputing Momentum strategy across crisis windows...")
    mom_data = {}
    for crisis_label, (c_start, c_end) in CRISIS_WINDOWS.items():
        points = []
        for pct in REDISTRIBUTION_LEVELS:
            ret, dd = run_strategy_crisis(
                prices, cash_prices,
                weight_type="momentum",
                redistribution_pct=pct,
                crisis_start=c_start,
                crisis_end=c_end,
                weight_kwargs=momentum_kwargs,
                initial_capital=initial_capital,
                slippage_bps=slippage_bps,
            )
            points.append((pct, ret, dd))
            logger.info("  %s / Mom pct=%.0f%%: Return=%.1f%%  MaxDD=%.1f%%",
                        crisis_label, pct * 100, ret * 100, dd * 100)
        mom_data[crisis_label] = points

    # ── Combined plot ───────────────────────────────────────────────────
    plot_crisis_scatter(
        ew_data, mom_data, benchmark_data,
        os.path.join(output_dir, "crisis_scatter.png"),
    )

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
