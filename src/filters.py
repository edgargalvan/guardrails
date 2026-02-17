"""Signal filter functions.

Each filter returns a boolean DataFrame (True = asset is "in", above trend).
Filters are looked up by name from the config via getattr().

Signature:
    def filter_name(prices: pd.DataFrame, **kwargs) -> pd.DataFrame

Adapted from trading/src/signals/trend.py.
"""

import pandas as pd

from . import data as data_module


# ===================================================================
# PUBLIC API — these are the names referenced in default.yaml
# ===================================================================

def sma(prices: pd.DataFrame, window: int = 200,
        frequency: str = "monthly", **kwargs) -> pd.DataFrame:
    """Simple moving average trend filter.

    Args:
        prices: Daily adjusted close prices, columns = tickers.
        window: Moving average lookback in trading days.
        frequency: How often to evaluate: "monthly", "weekly", "daily".
            Signal is evaluated at the specified frequency and
            forward-filled to daily resolution.

    Returns:
        Boolean DataFrame, same shape as prices. True = above trend.
    """
    tickers = list(prices.columns)
    above_daily = _compute_per_asset_trend(prices, tickers, window)

    if frequency == "monthly":
        return _evaluate_monthly(above_daily, prices)
    elif frequency == "weekly":
        return _evaluate_weekly(above_daily, prices)
    else:
        return above_daily


# ===================================================================
# INTERNAL HELPERS — adapted from trading/src/signals/trend.py
# ===================================================================

def _compute_per_asset_trend(prices: pd.DataFrame, tickers: list,
                             ma_period: int = 200) -> pd.DataFrame:
    """Compute daily above/below moving average flag for each asset.

    Args:
        prices: DataFrame of adjusted close prices.
        tickers: List of ticker symbols to compute trend for.
        ma_period: Moving average period (default 200).

    Returns:
        DataFrame of booleans (True = above MA = invested).
    """
    above_trend = pd.DataFrame(index=prices.index, columns=tickers, dtype=bool)

    for asset in tickers:
        ma = prices[asset].rolling(ma_period, min_periods=100).mean()
        above_trend[asset] = prices[asset] > ma

    # Fill warmup period (first ~ma_period days) as True (invested)
    above_trend = above_trend.fillna(True)
    return above_trend


def _evaluate_monthly(above_trend_daily: pd.DataFrame,
                      prices: pd.DataFrame) -> pd.DataFrame:
    """Evaluate trend on last trading day of each month, forward-fill.

    Args:
        above_trend_daily: DataFrame of daily trend booleans.
        prices: DataFrame of prices (used for month-end date detection).

    Returns:
        DataFrame of booleans, values only change monthly.
    """
    monthly_idx = data_module.month_end_dates(prices.index)

    above_trend_monthly = above_trend_daily.loc[
        above_trend_daily.index.isin(monthly_idx)
    ]

    above_trend_monthly = above_trend_monthly.reindex(
        above_trend_daily.index
    ).ffill().fillna(True)

    return above_trend_monthly


def _evaluate_weekly(above_trend_daily: pd.DataFrame,
                     prices: pd.DataFrame) -> pd.DataFrame:
    """Evaluate trend on last trading day of each week, forward-fill.

    Args:
        above_trend_daily: DataFrame of daily trend booleans.
        prices: DataFrame of prices (used for week-end date detection).

    Returns:
        DataFrame of booleans, values only change weekly.
    """
    weekly_idx = prices.groupby(
        [prices.index.isocalendar().year,
         prices.index.isocalendar().week]
    ).apply(lambda x: x.index[-1])

    above_trend_weekly = above_trend_daily.loc[
        above_trend_daily.index.isin(weekly_idx)
    ]

    above_trend_weekly = above_trend_weekly.reindex(
        above_trend_daily.index
    ).ffill().fillna(True)

    return above_trend_weekly
