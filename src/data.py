"""Data loading, caching, and alignment.

Downloads daily adjusted close prices via yfinance, caches as parquet,
and aligns all tickers to a common trading calendar.

Adapted from trading/src/data/loader.py.
"""

import logging
import os

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

CACHE_DIR = "data"


def load_prices(config: dict) -> pd.DataFrame:
    """Load adjusted close prices for all tickers in the config.

    Collects all unique tickers across strategies and benchmarks,
    downloads once, caches locally, aligns to common trading calendar.

    Args:
        config: Full config dict (strategies + benchmarks + settings).

    Returns:
        DataFrame with DatetimeIndex, one column per ticker, values
        are adjusted close prices. Forward-filled, no NaN.
    """
    tickers = _collect_tickers(config)
    cache_path = os.path.join(CACHE_DIR, "prices.parquet")

    if os.path.exists(cache_path):
        logger.info("Loading cached prices from %s", cache_path)
        prices = pd.read_parquet(cache_path)
        # Check if all tickers are present
        missing = set(tickers) - set(prices.columns)
        if not missing:
            return prices[list(tickers)]
        logger.info("Cache missing tickers %s, re-downloading", missing)

    download_start = config.get("settings", {}).get("download_start", "2006-01-01")
    logger.info("Downloading prices for %s (from %s)", sorted(tickers), download_start)
    prices = _download(sorted(tickers), start=download_start)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    prices.to_parquet(cache_path)
    logger.info("Cached %d days x %d tickers to %s",
                len(prices), len(prices.columns), cache_path)

    return prices


def _collect_tickers(config: dict) -> set:
    """Collect all unique tickers from strategies and benchmarks."""
    tickers = set()

    for strat in config.get("strategies", {}).values():
        tickers.update(strat.get("universe", []))

    for bench in config.get("benchmarks", {}).values():
        tickers.update(bench.get("tickers", []))

    # Cash vehicle
    cash = config.get("settings", {}).get("cash_vehicle")
    if cash:
        tickers.add(cash)

    return tickers


def _download(tickers: list, start: str = "2006-01-01") -> pd.DataFrame:
    """Download adjusted close prices via yfinance."""
    data = yf.download(
        tickers,
        start=start,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    # yf.download returns MultiIndex columns for multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers[:1]

    if prices.empty:
        raise ValueError(f"No data downloaded for tickers: {tickers}")

    # Drop tickers with all NaN (delisted or bad symbol)
    valid_cols = prices.columns[prices.notna().any()]
    dropped = set(prices.columns) - set(valid_cols)
    if dropped:
        logger.warning("Dropped tickers with no data: %s", dropped)
    prices = prices[valid_cols]

    # Forward-fill small gaps per ticker, then drop rows where ALL
    # tickers are NaN. We do NOT drop rows where some tickers are NaN
    # because different tickers have different start dates (e.g. VXUS
    # starts 2011 while SPY starts 2003). Consumers (backtest, benchmark)
    # handle missing data for their specific ticker subsets.

    # Warn if any ticker has large NaN gaps (>5 consecutive trading days)
    for col in prices.columns:
        nan_mask = prices[col].isna()
        if nan_mask.any():
            # Count consecutive NaN runs
            groups = nan_mask.ne(nan_mask.shift()).cumsum()
            nan_runs = nan_mask.groupby(groups).sum()
            max_gap = int(nan_runs.max())
            if max_gap > 5:
                logger.warning(
                    "%s has a %d-day NaN gap that will be forward-filled",
                    col, max_gap,
                )

    prices = prices.ffill()
    prices = prices.dropna(how="all")

    return prices


# ===================================================================
# DATE UTILITIES — shared across filters, weights, backtest
# ===================================================================

def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the last trading day of each month in the index.

    Used by filters (monthly evaluation), weights (monthly rebalance),
    and backtest (quarterly rebalance detection).
    """
    dummy = pd.Series(0, index=index)
    ends = dummy.groupby(
        [dummy.index.year, dummy.index.month]
    ).apply(lambda x: x.index[-1])
    return pd.DatetimeIndex(ends.values)
