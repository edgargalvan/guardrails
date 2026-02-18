"""Comparison orchestrator.

Runs all strategies and benchmarks across all configured time windows.
This is the core of the framework — it wires together data, filters,
weights, backtest, and metrics.

Not user-extensible. This is internal plumbing.
"""

import logging

import pandas as pd

from . import filters as filters_module
from . import weights as weights_module
from . import backtest
from . import metrics

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def run_comparison(config: dict, prices: pd.DataFrame) -> dict:
    """Run all strategies and benchmarks across all time windows.

    Strategies are computed once on the full price history, then the
    equity curve is trimmed per window. Benchmarks must be re-run per
    window because their start date affects the weight-drift simulation.

    Args:
        config: Full config dict from default.yaml.
        prices: DataFrame of daily adjusted close prices for all tickers.

    Returns:
        Nested dict:
        {
            window_label: {
                strategy_or_benchmark_name: {
                    "equity_curve": pd.Series,
                    "returns": pd.Series,
                    "metrics": {metric_name: value, ...}
                }
            }
        }
    """
    settings = config.get("settings", {})
    windows = settings.get("windows", [])
    metric_names = settings.get("metrics", ["sharpe", "cagr", "max_drawdown"])
    initial_capital = settings.get("initial_capital", 200_000)
    cash_vehicle = settings.get("cash_vehicle")
    benchmark_rebalance = settings.get("benchmark_rebalance", "quarterly")
    slippage_bps = settings.get("costs", {}).get("slippage_bps", 2.0)
    commission = settings.get("costs", {}).get("commission_per_trade", 0.0)

    # Cash prices
    cash_prices = prices[cash_vehicle] if cash_vehicle and cash_vehicle in prices.columns else None

    # SPY returns for capture ratios
    spy_returns = None
    if "SPY" in prices.columns:
        spy_returns = prices["SPY"].pct_change().iloc[1:]

    strategies = config.get("strategies", {})
    benchmarks = config.get("benchmarks", {})

    # --- Pre-compute all strategy equity curves once ---
    strategy_equities = {}
    for strat_name, strat_config in strategies.items():
        try:
            equity = _run_strategy(
                strat_name, strat_config, prices,
                cash_prices, initial_capital, slippage_bps, commission,
            )
            strategy_equities[strat_name] = equity
        except Exception as e:
            logger.error("Strategy %s failed: %s", strat_name, e)

    results = {}

    for window in windows:
        window_label = window["label"]
        window_start = pd.Timestamp(window["start"])
        logger.info("=== Window: %s (from %s) ===", window_label, window_start.date())

        window_results = {}

        # --- Trim pre-computed strategy equity curves to window ---
        for strat_name, equity in strategy_equities.items():
            equity_w = equity[equity.index >= window_start]
            if len(equity_w) < 63:
                logger.warning("Strategy %s has <63 days in window %s, skipping",
                               strat_name, window_label)
                continue

            returns = equity_w.pct_change().iloc[1:]
            bench_ret = spy_returns.reindex(returns.index) if spy_returns is not None else None
            m = metrics.compute_all(returns, equity_w, metric_names, bench_ret)

            window_results[strat_name] = {
                "equity_curve": equity_w,
                "returns": returns,
                "metrics": m,
            }
            logger.info("  %s: Sharpe=%.3f  CAGR=%.1f%%  MaxDD=%.1f%%",
                         strat_name,
                         m.get("sharpe", 0),
                         m.get("cagr", 0) * 100,
                         m.get("max_drawdown", 0) * 100)

        # --- Run benchmarks (must re-run per window for C3 fix) ---
        for bench_name, bench_config in benchmarks.items():
            try:
                allocation = bench_config["weights"]
                # Trim prices to window BEFORE calling run_benchmark.
                # Without this, late-starting tickers (e.g. VXUS ~2011)
                # cause dropna() inside run_benchmark to truncate all
                # windows to the same start date (C3 fix).
                bench_prices = prices[prices.index >= window_start]
                equity = backtest.run_benchmark(
                    bench_prices, allocation, initial_capital,
                    benchmark_rebalance, slippage_bps,
                )
            except Exception as e:
                logger.error("Benchmark %s failed: %s", bench_name, e)
                continue

            # Trim to window (prices were already sliced above, but
            # dropna() inside run_benchmark may shift the start forward
            # for late-starting tickers)
            equity = equity[equity.index >= window_start]
            if len(equity) < 63:
                continue

            # Warn if benchmark data starts significantly after window
            actual_start = equity.index[0]
            if actual_start > window_start + pd.Timedelta(days=30):
                logger.warning("  %s data starts %s, after window start %s",
                               bench_name, actual_start.date(),
                               window_start.date())

            returns = equity.pct_change().iloc[1:]
            bench_ret = spy_returns.reindex(returns.index) if spy_returns is not None else None
            m = metrics.compute_all(returns, equity, metric_names, bench_ret)

            window_results[bench_name] = {
                "equity_curve": equity,
                "returns": returns,
                "metrics": m,
            }
            logger.info("  %s: Sharpe=%.3f  CAGR=%.1f%%  MaxDD=%.1f%%",
                         bench_name,
                         m.get("sharpe", 0),
                         m.get("cagr", 0) * 100,
                         m.get("max_drawdown", 0) * 100)

        results[window_label] = window_results

    return results


def _run_strategy(
    name: str,
    strat_config: dict,
    all_prices: pd.DataFrame,
    cash_prices: pd.Series,
    initial_capital: float,
    slippage_bps: float,
    commission: float,
) -> pd.Series:
    """Run a single strategy on all available data.

    Returns equity curve from first available date (not windowed).
    Windowing is done by the caller.
    """
    universe = strat_config["universe"]
    exit_mode = strat_config.get("exit_mode", "cash")
    redistribution_pct = strat_config.get("redistribution_pct")

    # Slice prices to universe
    strat_prices = all_prices[universe].dropna()

    # Resolve and run filter
    filter_config = strat_config.get("filter", {}).copy()
    filter_type = filter_config.pop("type", "sma")
    filter_fn = getattr(filters_module, filter_type)
    in_mask = filter_fn(strat_prices, **filter_config)

    # Resolve and run weighting
    weight_config = strat_config.get("weights", {}).copy()
    weight_type = weight_config.pop("type", "equal")
    weight_fn = getattr(weights_module, weight_type)
    target_weights = weight_fn(strat_prices, in_mask, **weight_config)

    # Run backtest
    equity = backtest.run_backtest(
        prices=strat_prices,
        weights=target_weights,
        exit_mode=exit_mode,
        cash_prices=cash_prices,
        initial_capital=initial_capital,
        slippage_bps=slippage_bps,
        commission_per_trade=commission,
        redistribution_pct=redistribution_pct,
    )

    return equity
