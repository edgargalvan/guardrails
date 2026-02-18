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


def run_sweep(
    prices: pd.DataFrame,
    cash_prices: pd.Series,
    universe: list,
    weight_type: str = "equal",
    weight_kwargs: dict = None,
    window_start: str = "2007-07-01",
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


def plot_sweep(df: pd.DataFrame, title: str, output_path: str):
    """Create dual-axis plot: Sharpe on left, CAGR/MaxDD on right."""
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


def plot_sweep_scatter(df: pd.DataFrame, title: str, output_path: str):
    """Scatter plot: x=MaxDD, y=CAGR, color=redistribution_pct, size=Sharpe."""
    fig, ax = plt.subplots(figsize=(12, 8))

    x = df["max_drawdown"] * 100
    y = df["cagr"] * 100
    color = df["redistribution_pct"]
    # Scale Sharpe to visible marker sizes (min ~80, max ~400)
    sharpe_min, sharpe_max = df["sharpe"].min(), df["sharpe"].max()
    if sharpe_max - sharpe_min > 1e-6:
        size = 80 + 320 * (df["sharpe"] - sharpe_min) / (sharpe_max - sharpe_min)
    else:
        size = 200

    scatter = ax.scatter(x, y, c=color, s=size, cmap="RdYlGn_r",
                         edgecolors="black", linewidths=0.5, alpha=0.9,
                         vmin=0, vmax=1)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Redistribution to Survivors", fontsize=11)
    cbar.set_ticks([0, 0.25, 0.50, 0.75, 1.0])
    cbar.set_ticklabels(["0%\n(cash)", "25%", "50%", "75%", "100%\n(renorm)"])

    # Size legend
    for sharpe_val, label in [(sharpe_min, f"{sharpe_min:.2f}"),
                               (sharpe_max, f"{sharpe_max:.2f}")]:
        if sharpe_max - sharpe_min > 1e-6:
            s = 80 + 320 * (sharpe_val - sharpe_min) / (sharpe_max - sharpe_min)
        else:
            s = 200
        ax.scatter([], [], s=s, c="gray", edgecolors="black",
                   linewidths=0.5, label=f"Sharpe {label}")
    ax.legend(loc="upper left", fontsize=10, title="Marker Size",
              title_fontsize=10)

    # Label endpoints
    for idx in [0, len(df) - 1]:
        row = df.iloc[idx]
        pct_label = f"{row['redistribution_pct']*100:.0f}%"
        ax.annotate(pct_label,
                    (row["max_drawdown"] * 100, row["cagr"] * 100),
                    fontsize=9, fontweight="bold",
                    xytext=(8, -5), textcoords="offset points")

    ax.set_xlabel("Max Drawdown (%)", fontsize=12)
    ax.set_ylabel("CAGR (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved scatter: %s", output_path)


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
    )

    logger.info("\n" + "=" * 60)
    logger.info("  SWEEP COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
