"""Performance metrics.

Metrics are computed via an explicit dispatch table that calls quantstats
(qs.stats) for standard metrics and custom implementations for the rest.
The dispatch table uses a uniform (returns, equity_curve, benchmark_returns)
signature so compute_all() doesn't need inspect.signature() magic.

Public function aliases (sharpe, max_drawdown, etc.) are kept for backward
compatibility with tests and external callers.
"""

import logging

import numpy as np
import pandas as pd
import quantstats as qs

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

ANN_FACTOR = 252


def _safe(value: float) -> float:
    """Return 0.0 for None, NaN, Inf, or non-numeric values."""
    if value is None:
        return 0.0
    try:
        if np.isnan(value) or np.isinf(value):
            return 0.0
    except (TypeError, ValueError):
        return 0.0
    return float(value)


# ===================================================================
# CUSTOM METRIC IMPLEMENTATIONS (no qs equivalent)
# ===================================================================

def _max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a negative fraction.

    Direct computation from equity curve — no quantstats dependency.
    This avoids the M7 bug where qs.stats.max_drawdown auto-detects
    whether input is returns or prices.
    """
    if equity_curve is None or len(equity_curve) < 2:
        return 0.0
    if equity_curve.isna().all():
        return 0.0
    dd = equity_curve / equity_curve.cummax() - 1
    result = float(dd.min())
    return 0.0 if np.isnan(result) else result


def _worst_month(returns: pd.Series) -> float:
    """Worst monthly return (as simple return fraction)."""
    if len(returns) == 0:
        return 0.0
    monthly = returns.groupby(
        [returns.index.year, returns.index.month]
    ).apply(lambda x: (1 + x).prod() - 1)
    if len(monthly) == 0:
        return 0.0
    return float(monthly.min())


def _worst_year(returns: pd.Series) -> float:
    """Worst annual return (as simple return fraction)."""
    if len(returns) == 0:
        return 0.0
    annual = returns.groupby(returns.index.year).apply(
        lambda x: (1 + x).prod() - 1
    )
    if len(annual) == 0:
        return 0.0
    return float(annual.min())


def _upside_capture(returns: pd.Series,
                    benchmark_returns: pd.Series) -> float:
    """Upside capture ratio vs benchmark.

    Ratio of strategy's average return on days the benchmark is up,
    divided by the benchmark's average return on those days.
    """
    if benchmark_returns is None or len(returns) == 0:
        return 0.0
    common = returns.index.intersection(benchmark_returns.index)
    if len(common) == 0:
        return 0.0
    ret = returns.loc[common]
    bench = benchmark_returns.loc[common]
    up_days = bench > 0
    if up_days.sum() == 0:
        return 0.0
    strat_up = ret[up_days].mean()
    bench_up = bench[up_days].mean()
    if bench_up == 0:
        return 0.0
    return float(strat_up / bench_up)


def _downside_capture(returns: pd.Series,
                      benchmark_returns: pd.Series) -> float:
    """Downside capture ratio vs benchmark.

    Ratio of strategy's average return on days the benchmark is down,
    divided by the benchmark's average return on those days.
    Lower is better (less participation in losses).
    """
    if benchmark_returns is None or len(returns) == 0:
        return 0.0
    common = returns.index.intersection(benchmark_returns.index)
    if len(common) == 0:
        return 0.0
    ret = returns.loc[common]
    bench = benchmark_returns.loc[common]
    down_days = bench < 0
    if down_days.sum() == 0:
        return 0.0
    strat_down = ret[down_days].mean()
    bench_down = bench[down_days].mean()
    if bench_down == 0:
        return 0.0
    return float(strat_down / bench_down)


# ===================================================================
# DISPATCH TABLE — maps metric names to (returns, equity, benchmark) -> float
# ===================================================================

_METRIC_DISPATCH = {
    "sharpe": lambda r, eq, br: _safe(
        qs.stats.sharpe(r, rf=0.0, periods=ANN_FACTOR)
    ),
    "sortino": lambda r, eq, br: _safe(
        qs.stats.sortino(r, rf=0, periods=ANN_FACTOR)
    ),
    "cagr": lambda r, eq, br: _safe(
        qs.stats.cagr(r, rf=0.0)
    ),
    "annual_vol": lambda r, eq, br: _safe(
        qs.stats.volatility(r, periods=ANN_FACTOR)
    ),
    "max_drawdown": lambda r, eq, br: _max_drawdown(eq),
    "calmar": lambda r, eq, br: _safe(
        qs.stats.calmar(r)
    ),
    "ulcer_index": lambda r, eq, br: _safe(
        qs.stats.ulcer_index(r)
    ),
    "worst_month": lambda r, eq, br: _worst_month(r),
    "worst_year": lambda r, eq, br: _worst_year(r),
    "upside_capture": lambda r, eq, br: _upside_capture(r, br),
    "downside_capture": lambda r, eq, br: _downside_capture(r, br),
}


# ===================================================================
# PUBLIC API — backward-compatible function aliases
# ===================================================================

def sharpe(returns: pd.Series, risk_free: pd.Series = None,
           **kwargs) -> float:
    """Annualized Sharpe ratio."""
    if len(returns) == 0:
        return 0.0
    if risk_free is not None:
        returns = returns - risk_free.reindex(returns.index).fillna(0.0)
    return _safe(qs.stats.sharpe(returns, rf=0.0, periods=ANN_FACTOR))


def sortino(returns: pd.Series, **kwargs) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    if len(returns) == 0:
        return 0.0
    return _safe(qs.stats.sortino(returns, rf=0, periods=ANN_FACTOR))


def cagr(equity_curve: pd.Series, **kwargs) -> float:
    """Compound annual growth rate from equity curve."""
    if len(equity_curve) < 2:
        return 0.0
    returns = equity_curve.pct_change().dropna()
    if len(returns) == 0:
        return 0.0
    return _safe(qs.stats.cagr(returns, rf=0.0))


def annual_vol(returns: pd.Series, **kwargs) -> float:
    """Annualized volatility."""
    if len(returns) == 0:
        return 0.0
    return _safe(qs.stats.volatility(returns, periods=ANN_FACTOR))


def max_drawdown(equity_curve: pd.Series, **kwargs) -> float:
    """Maximum peak-to-trough drawdown as a negative fraction."""
    return _max_drawdown(equity_curve)


def calmar(returns: pd.Series, **kwargs) -> float:
    """Calmar ratio: annualized return / abs(max drawdown)."""
    if len(returns) == 0:
        return 0.0
    return _safe(qs.stats.calmar(returns))


def ulcer_index(equity_curve: pd.Series, **kwargs) -> float:
    """Ulcer Index: RMS of drawdown percentages."""
    if len(equity_curve) == 0:
        return 0.0
    returns = equity_curve.pct_change().dropna()
    if len(returns) == 0:
        return 0.0
    return _safe(qs.stats.ulcer_index(returns))


def worst_month(returns: pd.Series, **kwargs) -> float:
    """Worst monthly return."""
    return _worst_month(returns)


def worst_year(returns: pd.Series, **kwargs) -> float:
    """Worst annual return."""
    return _worst_year(returns)


def upside_capture(returns: pd.Series,
                   benchmark_returns: pd.Series = None,
                   **kwargs) -> float:
    """Upside capture ratio vs benchmark."""
    return _upside_capture(returns, benchmark_returns)


def downside_capture(returns: pd.Series,
                     benchmark_returns: pd.Series = None,
                     **kwargs) -> float:
    """Downside capture ratio vs benchmark."""
    return _downside_capture(returns, benchmark_returns)


# ===================================================================
# UTILITIES
# ===================================================================

def compute_all(returns: pd.Series, equity_curve: pd.Series,
                metric_names: list,
                benchmark_returns: pd.Series = None) -> dict:
    """Compute all named metrics.

    Args:
        returns: Daily simple returns.
        equity_curve: Daily equity curve.
        metric_names: List of metric function names to compute.
        benchmark_returns: Optional benchmark returns for capture ratios.

    Returns:
        Dict of {metric_name: value}.
    """
    results = {}
    for name in metric_names:
        fn = _METRIC_DISPATCH.get(name)
        if fn is None:
            logger.warning("Unknown metric: %s", name)
            continue

        try:
            results[name] = round(fn(returns, equity_curve,
                                     benchmark_returns), 4)
        except Exception as e:
            logger.warning("Metric %s failed: %s", name, e)
            results[name] = None

    return results


def bootstrap_sharpe_ci(returns: np.ndarray, n_bootstrap: int = 1000,
                        block_size: int = 5, ci: float = 0.95,
                        seed: int = 42) -> tuple:
    """Block bootstrap confidence interval for Sharpe ratio.

    Args:
        returns: Array of daily returns.
        n_bootstrap: Number of bootstrap samples.
        block_size: Block size for block bootstrap.
        ci: Confidence level.
        seed: Random seed.

    Returns:
        (lower_bound, upper_bound) of Sharpe ratio CI.
    """
    rng = np.random.RandomState(seed)
    n = len(returns)

    if n < block_size * 2:
        return (0.0, 0.0)

    n_blocks = -(-n // block_size)  # ceiling division
    sharpes = []

    for _ in range(n_bootstrap):
        block_starts = rng.randint(0, n - block_size, size=n_blocks)
        blocks = [returns[i:i + block_size] for i in block_starts]
        sample = np.concatenate(blocks)
        std = sample.std()
        if std > 0:
            sharpes.append(sample.mean() / std * np.sqrt(252))

    if not sharpes:
        return (0.0, 0.0)

    lo = np.percentile(sharpes, (1 - ci) / 2 * 100)
    hi = np.percentile(sharpes, (1 + ci) / 2 * 100)
    return (lo, hi)
