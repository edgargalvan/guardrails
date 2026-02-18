"""Backtest engine.

Uses vectorbt for portfolio construction (run_backtest) and a custom
simulation loop for benchmark weight-drift (run_benchmark).

run_backtest: takes weights + prices + redistribution_pct, builds a vectorbt
Portfolio with target-percent orders, returns an equity curve. The
redistribution_pct parameter (0.0–1.0) controls what happens when assets
are filtered out: 0.0 = all freed weight goes to cash (Faber's original),
1.0 = all freed weight redistributed to survivors (full renormalize).

run_benchmark: simulates passive buy-and-hold with periodic rebalancing
and natural weight drift between rebalance dates.
"""

import logging

import numpy as np
import pandas as pd
import vectorbt as vbt

from . import data as data_module

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def run_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    exit_mode: str = None,
    cash_prices: pd.Series = None,
    initial_capital: float = 200_000,
    commission_per_trade: float = 0.0,
    slippage_bps: float = 2.0,
    redistribution_pct: float = None,
) -> pd.Series:
    """Run a single-period backtest via vectorbt.

    Args:
        prices: Daily adjusted close prices, columns = tickers.
        weights: Target weights DataFrame (same shape as prices).
            Weights for "out" assets should already be zero (set by
            the weighting function). This function handles redistribution.
        exit_mode: DEPRECATED. Use redistribution_pct instead.
            "cash" maps to redistribution_pct=0.0,
            "renormalize" maps to redistribution_pct=1.0.
            Ignored if redistribution_pct is explicitly set.
        cash_prices: Daily prices for cash vehicle (e.g. SHY).
            If None, uninvested cash earns 0%.
        initial_capital: Starting portfolio value.
        commission_per_trade: Commission as fraction of trade value
            (0.001 = 10bps). Combined with slippage into a single
            per-order fee for vectorbt.
        slippage_bps: Slippage in basis points per side.
        redistribution_pct: Float 0.0–1.0. When assets are filtered out,
            this fraction of freed weight is redistributed proportionally
            to survivors. The remainder goes to cash.
            0.0 = pure cash exit (Faber's original).
            1.0 = full renormalize (100% invested at all times).
            0.5 = half to survivors, half to cash.

    Returns:
        Daily equity curve as a Series.
    """
    # Resolve exit_mode → redistribution_pct (backward compatibility)
    if redistribution_pct is None:
        if exit_mode == "renormalize":
            redistribution_pct = 1.0
        else:
            redistribution_pct = 0.0

    # Align weights to prices index
    w = weights.reindex(prices.index).ffill().fillna(0.0)

    # Apply partial redistribution
    # When assets are filtered out, their weight is zero (set by weight
    # functions). The "freed" weight is 1.0 - sum(survivor weights).
    # We redistribute redistribution_pct of that freed weight back to
    # survivors, proportionally to their current weights.
    row_sums = w.sum(axis=1)
    freed = (1.0 - row_sums).clip(lower=0.0)

    if redistribution_pct > 0:
        # Scale factor: each survivor's weight is multiplied by this.
        # At pct=1.0: scale = 1/row_sums (full renormalize).
        # At pct=0.5: scale = 1 + 0.5 * freed/row_sums.
        scale = 1.0 + (redistribution_pct * freed / row_sums.replace(0, np.nan))
        w = w.mul(scale.fillna(1.0), axis=0)

    # Cash vehicle: add explicit column so vectorbt invests the
    # uninvested portion in SHY instead of earning 0%.
    p = prices.copy()
    cash_col = (1.0 - w.sum(axis=1)).clip(lower=0.0)

    # Leverage warning
    max_weight_sum = w.sum(axis=1).max()
    if max_weight_sum > 1.0 + 1e-8:
        logger.warning(
            "Weights sum > 1.0 on some days (max: %.4f). "
            "Portfolio is leveraged.", max_weight_sum,
        )

    # Add cash column if any day has meaningful cash allocation.
    # Even at redistribution_pct=1.0, all-out days (all assets below
    # 200dma) have cash_col=1.0 and should earn SHY return.
    has_cash = float(cash_col.max()) > 1e-8
    if cash_prices is not None and has_cash:
        w = w.copy()
        w["_CASH"] = cash_col
        p["_CASH"] = cash_prices.reindex(p.index).ffill()
    # If no cash_prices or no cash allocation, uninvested earns 0%

    # Align indices
    common_idx = w.index.intersection(p.index)
    w = w.loc[common_idx]
    p = p[w.columns].loc[common_idx]

    # Combined cost: slippage + commission, applied per-order as a
    # fraction of trade value
    total_cost_rate = slippage_bps / 10_000 + commission_per_trade

    # Build portfolio via vectorbt
    pf = vbt.Portfolio.from_orders(
        close=p,
        size=w,
        size_type="targetpercent",
        group_by=True,
        cash_sharing=True,
        init_cash=float(initial_capital),
        fees=total_cost_rate,
        freq="1D",
    )

    equity = pf.value()
    equity.name = None
    return equity


def run_benchmark(
    prices: pd.DataFrame,
    allocation: dict,
    initial_capital: float = 200_000,
    rebalance: str = "quarterly",
    slippage_bps: float = 2.0,
) -> pd.Series:
    """Simulate a passive buy-and-hold benchmark with periodic rebalance.

    Weights drift between rebalance dates via simple return compounding.
    On rebalance dates, weights snap back to targets and turnover costs
    are charged.

    Args:
        prices: Daily adjusted close prices (must contain all tickers
            in allocation).
        allocation: Dict of {ticker: weight}, e.g. {"SPY": 0.60, "BND": 0.40}.
        initial_capital: Starting portfolio value.
        rebalance: "quarterly" (Mar/Jun/Sep/Dec) or "annual" or "monthly".
        slippage_bps: Slippage per rebalance event in basis points.

    Returns:
        Daily equity curve as a Series.
    """
    tickers = list(allocation.keys())
    target_weights = np.array([allocation[t] for t in tickers])

    weight_sum = target_weights.sum()
    if abs(weight_sum - 1.0) > 0.01:
        logger.warning(
            "Benchmark weights sum to %.4f (expected 1.0): %s",
            weight_sum, allocation,
        )

    port_prices = prices[tickers].dropna()
    simple_returns = port_prices / port_prices.shift(1) - 1
    simple_returns = simple_returns.iloc[1:]
    dates = simple_returns.index

    # Determine rebalance dates
    month_end_dates = set(data_module.month_end_dates(port_prices.index))

    if rebalance == "quarterly":
        rebal_months = {3, 6, 9, 12}
    elif rebalance == "annual":
        rebal_months = {12}
    elif rebalance == "monthly":
        rebal_months = set(range(1, 13))
    else:
        raise ValueError(
            f"Invalid rebalance frequency: {rebalance!r}. "
            f"Valid options: 'quarterly', 'annual', 'monthly'"
        )

    # Simulate with weight drift
    current_weights = target_weights.copy()
    daily_returns = []
    cost_per_unit = slippage_bps / 10_000

    for i in range(len(dates)):
        date = dates[i]
        day_ret = simple_returns.iloc[i].values

        port_ret = np.dot(current_weights, day_ret)
        daily_returns.append(port_ret)

        # Drift weights
        new_vals = current_weights * (1 + day_ret)
        total = new_vals.sum()
        if total > 0:
            current_weights = new_vals / total
        else:
            current_weights = target_weights.copy()

        # Rebalance on month-end
        if date in month_end_dates and date.month in rebal_months:
            # Cost of rebalancing
            turnover = np.abs(current_weights - target_weights).sum()
            daily_returns[-1] -= turnover * cost_per_unit
            current_weights = target_weights.copy()

    ret_series = pd.Series(daily_returns, index=dates, name="benchmark")
    equity = (1 + ret_series).cumprod() * initial_capital

    return equity
