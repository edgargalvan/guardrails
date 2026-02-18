#!/usr/bin/env python3
"""Sweep redistribution_pct from 0% to 100% and plot the results.

Runs the EW strategy with 200dma filter at 21 redistribution_pct values
(0.0 to 1.0 in steps of 0.05). For each value, computes Sharpe, CAGR,
and MaxDD. Outputs a CSV and a dual-axis plot.

Usage:
    python scripts/sweep_redistribution.py
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


# ── Benchmark definitions ────────────────────────────────────────────────────

BENCHMARKS = {
    "SPY": {"SPY": 1.0},
    "60/40": {"SPY": 0.60, "AGG": 0.40},
    "All-Weather-Lite": {"SPY": 0.30, "TLT": 0.40, "GLD": 0.15, "IEF": 0.15},
}

BENCHMARK_STYLES = {
    "SPY": {"color": "#888888", "linestyle": "--", "marker": "D"},
    "60/40": {"color": "#2E7D32", "linestyle": "-.", "marker": "s"},
    "All-Weather-Lite": {"color": "#FF8F00", "linestyle": ":", "marker": "^"},
}


def compute_benchmarks(
    prices: pd.DataFrame,
    window_start: str,
    window_end: str = None,
    initial_capital: float = 200_000,
    slippage_bps: float = 2.0,
) -> dict:
    """Compute Sharpe, CAGR, MaxDD for each benchmark in a given window.

    Returns dict of {bench_name: {"sharpe": ..., "cagr": ..., "max_drawdown": ...}}.
    """
    results = {}
    start_ts = pd.Timestamp(window_start)
    end_ts = pd.Timestamp(window_end) if window_end else None
    for bench_name, allocation in BENCHMARKS.items():
        # Check all tickers are available
        missing = [t for t in allocation if t not in prices.columns]
        if missing:
            logger.warning("  Benchmark %s: missing tickers %s, skipping",
                          bench_name, missing)
            continue

        equity = backtest.run_benchmark(
            prices=prices,
            allocation=allocation,
            initial_capital=initial_capital,
            rebalance="quarterly",
            slippage_bps=slippage_bps,
        )

        equity_w = equity[equity.index >= start_ts]
        if end_ts is not None:
            equity_w = equity_w[equity_w.index <= end_ts]
        if len(equity_w) < 63:
            continue

        returns = equity_w.pct_change().iloc[1:]
        m = metrics.compute_all(returns, equity_w,
                                ["sharpe", "cagr", "max_drawdown"])
        results[bench_name] = m
        logger.info("  Benchmark %s (%s): Sharpe=%.3f  CAGR=%.1f%%  MaxDD=%.1f%%",
                    bench_name, window_start, m["sharpe"],
                    m["cagr"] * 100, m["max_drawdown"] * 100)
    return results


def run_sweep(
    prices: pd.DataFrame,
    cash_prices: pd.Series,
    universe: list,
    weight_type: str = "equal",
    weight_kwargs: dict = None,
    window_start: str = "2007-07-01",
    window_end: str = None,
    initial_capital: float = 200_000,
    slippage_bps: float = 2.0,
    steps: int = 21,
) -> pd.DataFrame:
    """Run redistribution_pct sweep and return results DataFrame."""
    if weight_kwargs is None:
        weight_kwargs = {}

    strat_prices = prices[universe].dropna()

    # Compute filter and weights once (they don't change with redistribution_pct)
    in_mask = filters_module.sma(strat_prices, window=200, frequency="monthly")
    weight_fn = getattr(weights_module, weight_type)
    target_weights = weight_fn(strat_prices, in_mask, **weight_kwargs)

    pct_values = np.linspace(0.0, 1.0, steps)
    results = []

    for pct in pct_values:
        equity = backtest.run_backtest(
            prices=strat_prices,
            weights=target_weights,
            redistribution_pct=float(pct),
            cash_prices=cash_prices,
            initial_capital=initial_capital,
            slippage_bps=slippage_bps,
        )

        # Trim to window
        start_ts = pd.Timestamp(window_start)
        equity_w = equity[equity.index >= start_ts]
        if window_end is not None:
            end_ts = pd.Timestamp(window_end)
            equity_w = equity_w[equity_w.index <= end_ts]
        if len(equity_w) < 63:
            continue

        returns = equity_w.pct_change().iloc[1:]
        m = metrics.compute_all(returns, equity_w,
                                ["sharpe", "cagr", "max_drawdown"])

        results.append({
            "redistribution_pct": float(pct),
            "sharpe": m["sharpe"],
            "cagr": m["cagr"],
            "max_drawdown": m["max_drawdown"],
        })
        logger.info("  pct=%.2f  Sharpe=%.3f  CAGR=%.1f%%  MaxDD=%.1f%%",
                     pct, m["sharpe"], m["cagr"] * 100, m["max_drawdown"] * 100)

    return pd.DataFrame(results)


def plot_sweep(df: pd.DataFrame, title: str, output_path: str,
               bench_metrics: dict = None):
    """Create dual-axis plot: Sharpe on left, CAGR/MaxDD on right.

    Parameters
    ----------
    bench_metrics : dict, optional
        {bench_name: {"sharpe": ..., "cagr": ..., "max_drawdown": ...}}
        If provided, horizontal reference lines are drawn for each benchmark.
    """
    fig, ax1 = plt.subplots(figsize=(12, 7))

    x = df["redistribution_pct"] * 100  # Convert to percentage

    # Left axis: Sharpe
    color_sharpe = "#2E7D32"
    ax1.plot(x, df["sharpe"], color=color_sharpe, linewidth=2.5,
             marker="o", markersize=5, label="Sharpe Ratio")
    ax1.set_xlabel("Redistribution to Survivors (%)", fontsize=12)
    ax1.set_ylabel("Sharpe Ratio", color=color_sharpe, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color_sharpe)
    ax1.set_xlim(-2, 102)

    # Right axis: CAGR and MaxDD
    ax2 = ax1.twinx()
    color_cagr = "#1565C0"
    color_dd = "#FF5722"
    ax2.plot(x, df["cagr"] * 100, color=color_cagr, linewidth=2.5,
             marker="s", markersize=5, label="CAGR (%)")
    ax2.plot(x, df["max_drawdown"] * 100, color=color_dd, linewidth=2.5,
             marker="^", markersize=5, label="Max Drawdown (%)")
    ax2.set_ylabel("CAGR / Max Drawdown (%)", fontsize=12)

    # Benchmark reference points on the right margin
    if bench_metrics:
        bx = 105  # x-position just outside the sweep range
        for bench_name, bm in bench_metrics.items():
            style = BENCHMARK_STYLES.get(bench_name, {})
            bc = style.get("color", "#999999")
            mk = style.get("marker", "D")

            # Sharpe on left axis
            ax1.scatter(bx, bm["sharpe"], marker=mk, c=bc, s=80,
                        edgecolors="black", linewidths=0.5, alpha=0.9,
                        clip_on=False, zorder=5)
            ax1.annotate(f'{bench_name} {bm["sharpe"]:.2f}',
                         xy=(bx + 1, bm["sharpe"]),
                         fontsize=7, color=bc, fontweight="bold",
                         va="center", ha="left",
                         annotation_clip=False)

            # CAGR on right axis
            ax2.scatter(-5, bm["cagr"] * 100, marker=mk, c=bc, s=80,
                        edgecolors="black", linewidths=0.5, alpha=0.9,
                        clip_on=False, zorder=5)
            ax2.annotate(f'{bench_name} {bm["cagr"]*100:.1f}%',
                         xy=(-6, bm["cagr"] * 100),
                         fontsize=7, color=bc, fontweight="bold",
                         va="center", ha="right",
                         annotation_clip=False)

            # MaxDD on right axis
            ax2.scatter(-5, bm["max_drawdown"] * 100, marker=mk, c=bc, s=80,
                        edgecolors="black", linewidths=0.5, alpha=0.9,
                        clip_on=False, zorder=5)
            ax2.annotate(f'{bench_name} {bm["max_drawdown"]*100:.1f}%',
                         xy=(-6, bm["max_drawdown"] * 100),
                         fontsize=7, color=bc, fontweight="bold",
                         va="center", ha="right",
                         annotation_clip=False)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left",
               fontsize=10)

    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", output_path)


def plot_sweep_scatter(df: pd.DataFrame, title: str, output_path: str,
                       bench_metrics: dict = None):
    """Scatter plot: x=MaxDD, y=CAGR, color=redistribution_pct.

    Sharpe range noted as subtitle text (not encoded as dot size).
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    x = df["max_drawdown"] * 100
    y = df["cagr"] * 100
    color = df["redistribution_pct"]

    scatter = ax.scatter(x, y, c=color, s=120, cmap="RdYlGn_r",
                         edgecolors="black", linewidths=0.5, alpha=0.9,
                         vmin=0, vmax=1, zorder=3)

    # Thin connecting line
    ax.plot(x, y, color="black", linewidth=0.8, alpha=0.4, zorder=1)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Redistribution to Survivors", fontsize=11)
    cbar.set_ticks([0, 0.25, 0.50, 0.75, 1.0])
    cbar.set_ticklabels(["0%\n(cash)", "25%", "50%", "75%", "100%\n(renorm)"])

    # Sharpe range as subtitle
    sharpe_min, sharpe_max = df["sharpe"].min(), df["sharpe"].max()
    sharpe_mean = df["sharpe"].mean()
    sharpe_half_range = (sharpe_max - sharpe_min) / 2
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.text(0.5, 1.01, f"Sharpe: {sharpe_mean:.2f} \u00b1 {sharpe_half_range:.2f} across sweep",
            transform=ax.transAxes, fontsize=10, color="gray",
            ha="center", va="bottom")

    # Benchmark markers
    if bench_metrics:
        for bench_name, bm in bench_metrics.items():
            style = BENCHMARK_STYLES.get(bench_name, {})
            bc = style.get("color", "#999999")
            mk = style.get("marker", "D")
            ax.scatter(bm["max_drawdown"] * 100, bm["cagr"] * 100,
                       marker=mk, c=bc, s=160, edgecolors="black",
                       linewidths=0.5, alpha=0.9, zorder=4)
            ax.annotate(bench_name,
                        (bm["max_drawdown"] * 100, bm["cagr"] * 100),
                        fontsize=8, fontweight="bold", color=bc,
                        xytext=(7, 5), textcoords="offset points", zorder=5)

    # Label endpoints
    for idx in [0, len(df) - 1]:
        row = df.iloc[idx]
        pct_label = f"{row['redistribution_pct']*100:.0f}%"
        ax.annotate(pct_label,
                    (row["max_drawdown"] * 100, row["cagr"] * 100),
                    fontsize=9, fontweight="bold",
                    xytext=(8, -5), textcoords="offset points", zorder=5)

    ax.set_xlabel("Max Drawdown (%)", fontsize=12)
    ax.set_ylabel("CAGR (%)", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved scatter: %s", output_path)


def plot_sweep_overlay(series: list, title: str, output_path: str,
                       bench_metrics: dict = None):
    """Overlay multiple sweep series on one scatter plot.

    Color distinguishes series. Thin black line connects dots within each series.
    Sharpe range noted as subtitle text.

    Parameters
    ----------
    series : list of dict
        Each dict has keys: "df" (DataFrame), "label" (str), "color" (str).
    title : str
    output_path : str
    bench_metrics : dict, optional
        {bench_name: {"sharpe": ..., "cagr": ..., "max_drawdown": ...}}
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Collect global Sharpe range for subtitle
    all_sharpe = pd.concat([s["df"]["sharpe"] for s in series])
    sharpe_min, sharpe_max = all_sharpe.min(), all_sharpe.max()
    sharpe_mean = all_sharpe.mean()
    sharpe_half_range = (sharpe_max - sharpe_min) / 2

    for s in series:
        df = s["df"]
        c = s["color"]
        label = s["label"]

        x = df["max_drawdown"] * 100
        y = df["cagr"] * 100

        # Thin connecting line
        ax.plot(x, y, color="black", linewidth=0.8, alpha=0.5, zorder=1)

        # Scatter with uniform dot size
        ax.scatter(x, y, c=c, s=100, edgecolors="black", linewidths=0.4,
                   alpha=0.85, zorder=2, label=label)

        # Label endpoints (0% and 100%)
        for idx in [0, len(df) - 1]:
            row = df.iloc[idx]
            pct_label = f"{row['redistribution_pct']*100:.0f}%"
            ax.annotate(pct_label,
                        (row["max_drawdown"] * 100, row["cagr"] * 100),
                        fontsize=8, fontweight="bold", color=c,
                        xytext=(8, -4), textcoords="offset points",
                        zorder=3)

    # Benchmark markers
    if bench_metrics:
        for bench_name, bm in bench_metrics.items():
            style = BENCHMARK_STYLES.get(bench_name, {})
            bc = style.get("color", "#999999")
            mk = style.get("marker", "D")
            ax.scatter(bm["max_drawdown"] * 100, bm["cagr"] * 100,
                       marker=mk, c=bc, s=160, edgecolors="black",
                       linewidths=0.5, alpha=0.9, zorder=4)
            ax.annotate(bench_name,
                        (bm["max_drawdown"] * 100, bm["cagr"] * 100),
                        fontsize=8, fontweight="bold", color=bc,
                        xytext=(7, 5), textcoords="offset points", zorder=5)

    ax.legend(fontsize=10, loc="upper right")
    ax.set_xlabel("Max Drawdown (%)", fontsize=12)
    ax.set_ylabel("CAGR (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.text(0.5, 1.01, f"Sharpe: {sharpe_mean:.2f} \u00b1 {sharpe_half_range:.2f} across all points",
            transform=ax.transAxes, fontsize=10, color="gray",
            ha="center", va="bottom")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved overlay: %s", output_path)


# Color palette for time windows
WINDOW_COLORS = {
    "Full Period": "#1565C0",     # blue
    "Post-GFC": "#2E7D32",        # green
    "All Tickers": "#FF8F00",     # amber
    "Pre-COVID": "#7B1FA2",       # purple
    "Post-COVID": "#D32F2F",      # red
    "GFC-to-COVID": "#00838F",    # teal
}

# Colors for EW vs Momentum
EW_COLOR = "#1565C0"        # blue
MOM_COLOR = "#D32F2F"       # red


def main():
    # Load config and prices
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

    # === Compute benchmarks for each window ===
    # Windows: (start, end_or_None)
    window_defs = {
        "Full Period": ("2007-07-01", None),
        "Post-GFC": ("2009-04-01", None),
        "All Tickers": ("2011-02-01", None),
        "Pre-COVID": ("2018-01-01", None),
        "Post-COVID": ("2021-01-01", None),
        "GFC-to-COVID": ("2009-04-01", "2019-12-31"),
    }
    all_bench = {}
    for wlabel, (wstart, wend) in window_defs.items():
        logger.info("\n--- Benchmarks for %s (%s to %s) ---",
                    wlabel, wstart, wend or "present")
        all_bench[wlabel] = compute_benchmarks(
            prices, wstart, window_end=wend,
            initial_capital=initial_capital,
            slippage_bps=slippage_bps,
        )

    # === Primary sweep: EW, Full Period ===
    logger.info("\n" + "=" * 60)
    logger.info("  SWEEP: Equal Weight, Full Period (2007-07-01)")
    logger.info("=" * 60)
    df_ew = run_sweep(
        prices, cash_prices,
        universe=["SPY", "TLT", "GLD"],
        weight_type="equal",
        window_start="2007-07-01",
        initial_capital=initial_capital,
        slippage_bps=slippage_bps,
    )
    csv_path = os.path.join(output_dir, "redistribution_sweep.csv")
    df_ew.to_csv(csv_path, index=False)
    logger.info("Saved CSV: %s", csv_path)

    plot_sweep(
        df_ew,
        "Effect of Redistribution on EW Strategy (SPY/TLT/GLD, 200dma, Full Period)",
        os.path.join(output_dir, "redistribution_sweep.png"),
    )
    plot_sweep_scatter(
        df_ew,
        "Redistribution Dial: Risk vs Return (EW, Full Period)",
        os.path.join(output_dir, "redistribution_scatter.png"),
        bench_metrics=all_bench["Full Period"],
    )

    # === Secondary sweep: Momentum, Full Period ===
    logger.info("\n" + "=" * 60)
    logger.info("  SWEEP: Momentum Tilt, Full Period (2007-07-01)")
    logger.info("=" * 60)
    df_mom = run_sweep(
        prices, cash_prices,
        universe=["SPY", "TLT", "GLD"],
        weight_type="momentum",
        weight_kwargs={
            "lookback_days": 252,
            "skip_days": 21,
            "split": [0.70, 0.20, 0.10],
        },
        window_start="2007-07-01",
        initial_capital=initial_capital,
        slippage_bps=slippage_bps,
    )
    csv_path = os.path.join(output_dir, "redistribution_sweep_momentum.csv")
    df_mom.to_csv(csv_path, index=False)
    logger.info("Saved CSV: %s", csv_path)

    plot_sweep(
        df_mom,
        "Effect of Redistribution on Momentum Strategy (SPY/TLT/GLD, 200dma, Full Period)",
        os.path.join(output_dir, "redistribution_sweep_momentum.png"),
    )
    plot_sweep_scatter(
        df_mom,
        "Redistribution Dial: Risk vs Return (Momentum, Full Period)",
        os.path.join(output_dir, "redistribution_scatter_momentum.png"),
        bench_metrics=all_bench["Full Period"],
    )

    # === Secondary sweep: EW, Post-COVID ===
    logger.info("\n" + "=" * 60)
    logger.info("  SWEEP: Equal Weight, Post-COVID (2021-01-01)")
    logger.info("=" * 60)
    df_post = run_sweep(
        prices, cash_prices,
        universe=["SPY", "TLT", "GLD"],
        weight_type="equal",
        window_start="2021-01-01",
        initial_capital=initial_capital,
        slippage_bps=slippage_bps,
    )
    csv_path = os.path.join(output_dir, "redistribution_sweep_postcovid.csv")
    df_post.to_csv(csv_path, index=False)
    logger.info("Saved CSV: %s", csv_path)

    plot_sweep(
        df_post,
        "Effect of Redistribution on EW Strategy (SPY/TLT/GLD, 200dma, Post-COVID)",
        os.path.join(output_dir, "redistribution_sweep_postcovid.png"),
    )
    plot_sweep_scatter(
        df_post,
        "Redistribution Dial: Risk vs Return (EW, Post-COVID)",
        os.path.join(output_dir, "redistribution_scatter_postcovid.png"),
        bench_metrics=all_bench["Post-COVID"],
    )

    # === Secondary sweep: Momentum, Post-COVID ===
    logger.info("\n" + "=" * 60)
    logger.info("  SWEEP: Momentum Tilt, Post-COVID (2021-01-01)")
    logger.info("=" * 60)
    df_mom_post = run_sweep(
        prices, cash_prices,
        universe=["SPY", "TLT", "GLD"],
        weight_type="momentum",
        weight_kwargs={
            "lookback_days": 252,
            "skip_days": 21,
            "split": [0.70, 0.20, 0.10],
        },
        window_start="2021-01-01",
        initial_capital=initial_capital,
        slippage_bps=slippage_bps,
    )
    csv_path = os.path.join(output_dir, "redistribution_sweep_momentum_postcovid.csv")
    df_mom_post.to_csv(csv_path, index=False)
    logger.info("Saved CSV: %s", csv_path)

    plot_sweep(
        df_mom_post,
        "Effect of Redistribution on Momentum Strategy (SPY/TLT/GLD, 200dma, Post-COVID)",
        os.path.join(output_dir, "redistribution_sweep_momentum_postcovid.png"),
    )
    plot_sweep_scatter(
        df_mom_post,
        "Redistribution Dial: Risk vs Return (Momentum, Post-COVID)",
        os.path.join(output_dir, "redistribution_scatter_momentum_postcovid.png"),
        bench_metrics=all_bench["Post-COVID"],
    )

    # === GFC-to-COVID sweep (2009-04 to 2019-12): the long bull ===
    logger.info("\n" + "=" * 60)
    logger.info("  SWEEP: Equal Weight, GFC-to-COVID (2009-04 to 2019-12)")
    logger.info("=" * 60)
    df_ew_gfc2covid = run_sweep(
        prices, cash_prices,
        universe=["SPY", "TLT", "GLD"],
        weight_type="equal",
        window_start="2009-04-01",
        window_end="2019-12-31",
        initial_capital=initial_capital,
        slippage_bps=slippage_bps,
    )
    csv_path = os.path.join(output_dir, "redistribution_sweep_gfc_to_covid.csv")
    df_ew_gfc2covid.to_csv(csv_path, index=False)
    logger.info("Saved CSV: %s", csv_path)

    plot_sweep(
        df_ew_gfc2covid,
        "Effect of Redistribution on EW Strategy (SPY/TLT/GLD, 200dma, GFC-to-COVID)",
        os.path.join(output_dir, "redistribution_sweep_gfc_to_covid.png"),
    )
    plot_sweep_scatter(
        df_ew_gfc2covid,
        "Redistribution Dial: Risk vs Return (EW, GFC-to-COVID)",
        os.path.join(output_dir, "redistribution_scatter_gfc_to_covid.png"),
        bench_metrics=all_bench["GFC-to-COVID"],
    )

    logger.info("\n" + "=" * 60)
    logger.info("  SWEEP: Momentum Tilt, GFC-to-COVID (2009-04 to 2019-12)")
    logger.info("=" * 60)
    df_mom_gfc2covid = run_sweep(
        prices, cash_prices,
        universe=["SPY", "TLT", "GLD"],
        weight_type="momentum",
        weight_kwargs={
            "lookback_days": 252,
            "skip_days": 21,
            "split": [0.70, 0.20, 0.10],
        },
        window_start="2009-04-01",
        window_end="2019-12-31",
        initial_capital=initial_capital,
        slippage_bps=slippage_bps,
    )
    csv_path = os.path.join(output_dir, "redistribution_sweep_momentum_gfc_to_covid.csv")
    df_mom_gfc2covid.to_csv(csv_path, index=False)
    logger.info("Saved CSV: %s", csv_path)

    plot_sweep(
        df_mom_gfc2covid,
        "Effect of Redistribution on Momentum Strategy (SPY/TLT/GLD, 200dma, GFC-to-COVID)",
        os.path.join(output_dir, "redistribution_sweep_momentum_gfc_to_covid.png"),
    )
    plot_sweep_scatter(
        df_mom_gfc2covid,
        "Redistribution Dial: Risk vs Return (Momentum, GFC-to-COVID)",
        os.path.join(output_dir, "redistribution_scatter_momentum_gfc_to_covid.png"),
        bench_metrics=all_bench["GFC-to-COVID"],
    )

    # === Additional window sweeps for overlay plots ===
    momentum_kwargs = {
        "lookback_days": 252,
        "skip_days": 21,
        "split": [0.70, 0.20, 0.10],
    }
    extra_windows = [
        ("Post-GFC", "2009-04-01"),
        ("All Tickers", "2011-02-01"),
        ("Pre-COVID", "2018-01-01"),
    ]
    # Store all sweeps by (weight_type, window_label)
    all_sweeps = {
        ("equal", "Full Period"): df_ew,
        ("equal", "Post-COVID"): df_post,
        ("equal", "GFC-to-COVID"): df_ew_gfc2covid,
        ("momentum", "Full Period"): df_mom,
        ("momentum", "Post-COVID"): df_mom_post,
        ("momentum", "GFC-to-COVID"): df_mom_gfc2covid,
    }
    for label, start in extra_windows:
        logger.info("\n" + "=" * 60)
        logger.info("  SWEEP: Equal Weight, %s (%s)", label, start)
        logger.info("=" * 60)
        df_ew_w = run_sweep(
            prices, cash_prices,
            universe=["SPY", "TLT", "GLD"],
            weight_type="equal",
            window_start=start,
            initial_capital=initial_capital,
            slippage_bps=slippage_bps,
        )
        all_sweeps[("equal", label)] = df_ew_w

        logger.info("\n" + "=" * 60)
        logger.info("  SWEEP: Momentum Tilt, %s (%s)", label, start)
        logger.info("=" * 60)
        df_mom_w = run_sweep(
            prices, cash_prices,
            universe=["SPY", "TLT", "GLD"],
            weight_type="momentum",
            weight_kwargs=momentum_kwargs,
            window_start=start,
            initial_capital=initial_capital,
            slippage_bps=slippage_bps,
        )
        all_sweeps[("momentum", label)] = df_mom_w

    # === Overlay plots ===
    window_labels = ["Full Period", "Post-GFC", "All Tickers", "Pre-COVID",
                      "Post-COVID", "GFC-to-COVID"]

    # 1. EW: Full vs Post-COVID
    plot_sweep_overlay(
        [
            {"df": all_sweeps[("equal", "Full Period")], "label": "EW Full Period", "color": WINDOW_COLORS["Full Period"]},
            {"df": all_sweeps[("equal", "Post-COVID")], "label": "EW Post-COVID", "color": WINDOW_COLORS["Post-COVID"]},
        ],
        "EW Redistribution: Full Period vs Post-COVID",
        os.path.join(output_dir, "overlay_ew_full_vs_postcovid.png"),
        bench_metrics=all_bench["Full Period"],
    )

    # 2. Momentum: Full vs Post-COVID
    plot_sweep_overlay(
        [
            {"df": all_sweeps[("momentum", "Full Period")], "label": "Mom Full Period", "color": WINDOW_COLORS["Full Period"]},
            {"df": all_sweeps[("momentum", "Post-COVID")], "label": "Mom Post-COVID", "color": WINDOW_COLORS["Post-COVID"]},
        ],
        "Momentum Redistribution: Full Period vs Post-COVID",
        os.path.join(output_dir, "overlay_mom_full_vs_postcovid.png"),
        bench_metrics=all_bench["Full Period"],
    )

    # 3. Full Period: EW vs Momentum
    plot_sweep_overlay(
        [
            {"df": all_sweeps[("equal", "Full Period")], "label": "EW", "color": EW_COLOR},
            {"df": all_sweeps[("momentum", "Full Period")], "label": "Momentum", "color": MOM_COLOR},
        ],
        "Full Period: EW vs Momentum Redistribution",
        os.path.join(output_dir, "overlay_full_ew_vs_mom.png"),
        bench_metrics=all_bench["Full Period"],
    )

    # 4. Everything
    plot_sweep_overlay(
        [
            {"df": all_sweeps[("equal", "Full Period")], "label": "EW Full", "color": "#1565C0"},
            {"df": all_sweeps[("equal", "Post-COVID")], "label": "EW Post-COVID", "color": "#64B5F6"},
            {"df": all_sweeps[("momentum", "Full Period")], "label": "Mom Full", "color": "#D32F2F"},
            {"df": all_sweeps[("momentum", "Post-COVID")], "label": "Mom Post-COVID", "color": "#EF9A9A"},
        ],
        "All Sweeps: EW vs Momentum \u00d7 Full vs Post-COVID",
        os.path.join(output_dir, "overlay_all.png"),
        bench_metrics=all_bench["Full Period"],
    )

    # 5. EW: all time windows
    plot_sweep_overlay(
        [
            {"df": all_sweeps[("equal", w)], "label": f"EW {w}", "color": WINDOW_COLORS[w]}
            for w in window_labels
        ],
        "EW Redistribution Across Investment Horizons",
        os.path.join(output_dir, "overlay_ew_all_windows.png"),
        bench_metrics=all_bench["Full Period"],
    )

    # 6. Momentum: all time windows
    plot_sweep_overlay(
        [
            {"df": all_sweeps[("momentum", w)], "label": f"Mom {w}", "color": WINDOW_COLORS[w]}
            for w in window_labels
        ],
        "Momentum Redistribution Across Investment Horizons",
        os.path.join(output_dir, "overlay_mom_all_windows.png"),
        bench_metrics=all_bench["Full Period"],
    )

    logger.info("\n" + "=" * 60)
    logger.info("  SWEEP COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
