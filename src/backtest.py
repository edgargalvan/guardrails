"""Backtest engine.

Uses vectorbt for portfolio construction (run_backtest) and a custom
simulation loop for benchmark weight-drift (run_benchmark).

run_backtest: takes weights + prices + exit mode, builds a vectorbt
Portfolio with target-percent orders, returns an equity curve.

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
    exit_mode: str,
    cash_prices: pd.Series = None,
    initial_capital: float = 200_000,
    commission_per_trade: float = 0.0,
    slippage_bps: float = 2.0,
) -> pd.Series:
    """Run a single-period backtest via vectorbt.

    Args:
        prices: Daily adjusted close prices, columns = tickers.
        weights: Target weights DataFrame (same shape as prices).
            Weights for "out" assets should already be zero (set by
            the weighting function). This function handles exit_mode.
        exit_mode: "cash" or "renormalize".
            - cash: survivors keep computed weights, remainder is invested
              in the cash vehicle (e.g. SHY). Total risky exposure drops
              when assets are filtered out.
            - renormalize: survivors' weights scaled to sum to 1.0.
              Stay ~100% invested at all times.
        cash_prices: Daily prices for cash vehicle (e.g. SHY).
            Used when exit_mode == "cash". If None, cash earns 0%.
        initial_capital: Starting portfolio value.
        commission_per_trade: Commission as fraction of trade value
            (0.001 = 10bps). Combined with slippage into a single
            per-order fee for vectorbt.
        slippage_bps: Slippage in basis points per side.

    Returns:
        Daily equity curve as a Series.
    """
    # Align weights to prices index
    w = weights.reindex(prices.index).ffill().fillna(0.0)

    # Apply exit mode
    if exit_mode == "renormalize":
        row_sums = w.sum(axis=1)
        w = w.div(row_sums.replace(0, np.nan), axis=0).fillna(0.0)

    # Cash vehicle: add as explicit column so vectorbt invests the
    # uninvested portion in SHY instead of earning 0%.
    p = prices.copy()
    if exit_mode == "cash":
        max_weight_sum = w.sum(axis=1).max()
        if max_weight_sum > 1.0 + 1e-8:
            logger.warning(
                "Weights sum > 1.0 on some days (max: %.4f). "
                "Portfolio is leveraged.", max_weight_sum,
            )
        cash_col = (1.0 - w.sum(axis=1)).clip(lower=0.0)
        if cash_prices is not None:
            w = w.copy()
            w["_CASH"] = cash_col
            p["_CASH"] = cash_prices.reindex(p.index).ffill()
        # If no cash_prices, uninvested portion earns 0% (vectorbt default)

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
