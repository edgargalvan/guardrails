"""Tests for src/filters.py.

Ported from trading/scripts/integrity_tests.py (II.1, II.3)
plus new structural validation tests.
"""

import numpy as np
import pandas as pd
import pytest

from src import filters


# ===================================================================
# Structural tests
# ===================================================================

def test_sma_returns_boolean_dataframe(synthetic_rising_prices):
    """sma() output must be a boolean DataFrame matching input shape."""
    prices = synthetic_rising_prices
    result = filters.sma(prices, window=200, frequency="monthly")

    assert isinstance(result, pd.DataFrame)
    assert result.shape == prices.shape
    assert list(result.columns) == list(prices.columns)
    # All values should be boolean-like (True/False).
    # Note: fillna(True) in filters.py may produce object dtype
    unique_vals = set()
    for col in result.columns:
        unique_vals.update(result[col].unique())
    assert unique_vals <= {True, False}, \
        f"Expected only True/False values, got: {unique_vals}"


def test_sma_monthly_forward_filled(synthetic_rising_prices):
    """Monthly filter signal should only change on month-end dates."""
    prices = synthetic_rising_prices
    result = filters.sma(prices, window=200, frequency="monthly")

    # Find dates where the signal changes
    changes = result.astype(int).diff().abs().sum(axis=1)
    change_dates = changes[changes > 0].index

    # Get month-end trading dates
    month_ends = prices.groupby(
        [prices.index.year, prices.index.month]
    ).apply(lambda x: x.index[-1])

    # All change dates should be on or 1 day after a month-end evaluation
    # (forward-fill means the change appears on the next trading day after
    # the month-end evaluation, which IS the month-end date itself since
    # evaluation happens at close and is applied starting that day)
    for dt in change_dates:
        # The change should be on a month-end date or the first trading
        # day after one (due to forward fill boundary)
        is_month_end = dt in month_ends.values
        # Allow 1 day tolerance for month boundaries
        pos = prices.index.get_loc(dt)
        if pos > 0:
            prev = prices.index[pos - 1]
            is_day_after_month_end = prev in month_ends.values
        else:
            is_day_after_month_end = False

        assert is_month_end or is_day_after_month_end, \
            f"Signal changed on {dt.date()} which is not a month boundary"


# ===================================================================
# Deterministic tests (ported from integrity II.1)
# ===================================================================

def test_sma_all_above_after_warmup(synthetic_rising_prices):
    """Linearly rising prices must be above 200dma after warmup.

    Ported from integrity_tests.py II.1: deterministic 200dma.
    """
    prices = synthetic_rising_prices
    result = filters.sma(prices, window=200, frequency="monthly")

    # After 250 days (generous warmup), all assets should be True
    post_warmup = result.iloc[250:]
    assert post_warmup.all().all(), \
        f"Found False values after warmup: {(~post_warmup).sum().to_dict()}"


def test_sma_all_below_after_warmup(synthetic_falling_prices):
    """Linearly falling prices must be below 200dma after warmup."""
    prices = synthetic_falling_prices
    result = filters.sma(prices, window=200, frequency="monthly")

    # After 250 days, all assets should be False
    post_warmup = result.iloc[250:]
    assert (~post_warmup).all().all(), \
        f"Found True values after warmup: {post_warmup.sum().to_dict()}"


# ===================================================================
# Look-ahead bias test (ported from integrity II.3)
# ===================================================================

def test_sma_signal_stable_under_output_shift(synthetic_rising_prices):
    """Shifting the output signal by 1 day should not change most values.

    Ported from integrity_tests.py II.3.
    This verifies signal stability: a one-day shift of the output should
    differ on < 5% of days. NOTE: this tests output stability, not true
    look-ahead bias (see test_system.py::test_look_ahead_bias_ma_shift
    for the proper input-shift test).
    """
    prices = synthetic_rising_prices

    # Normal filter
    result_normal = filters.sma(prices, window=200, frequency="monthly")

    # Shift the output by 1 day to measure signal stability
    result_shifted = result_normal.shift(1).fillna(True)

    # Count differing days (after warmup to avoid startup artifacts)
    post_warmup_normal = result_normal.iloc[250:]
    post_warmup_shifted = result_shifted.reindex(post_warmup_normal.index).fillna(True)

    n_total = len(post_warmup_normal) * len(post_warmup_normal.columns)
    n_differ = (post_warmup_normal != post_warmup_shifted).sum().sum()
    pct_differ = n_differ / n_total

    assert pct_differ < 0.05, \
        f"Signal differs on {pct_differ:.1%} of days (> 5% threshold)"


# ===================================================================
# Frequency tests (G2)
# ===================================================================

def test_sma_weekly_returns_boolean(synthetic_rising_prices):
    """Weekly frequency filter should return a boolean DataFrame."""
    prices = synthetic_rising_prices
    result = filters.sma(prices, window=200, frequency="weekly")

    assert isinstance(result, pd.DataFrame)
    assert result.shape == prices.shape
    unique_vals = set()
    for col in result.columns:
        unique_vals.update(result[col].unique())
    assert unique_vals <= {True, False}


def test_sma_daily_no_forward_fill(synthetic_rising_prices):
    """Daily frequency should evaluate every day (no forward-fill grouping).

    After warmup, rising prices should be above SMA every single day.
    """
    prices = synthetic_rising_prices
    result = filters.sma(prices, window=200, frequency="daily")

    assert isinstance(result, pd.DataFrame)
    # After warmup, all rising prices should be True every day
    post_warmup = result.iloc[250:]
    assert post_warmup.all().all(), \
        f"Daily filter on rising prices has False values: {(~post_warmup).sum().to_dict()}"


def test_sma_weekly_changes_only_at_week_ends(synthetic_rising_prices):
    """Weekly filter signal should only change at week-end dates."""
    prices = synthetic_rising_prices
    result = filters.sma(prices, window=200, frequency="weekly")

    # Signal should be constant within each week (only change at week boundaries)
    post_warmup = result.iloc[250:]
    changes = post_warmup.astype(int).diff().abs().sum(axis=1)
    change_dates = changes[changes > 0].index

    # Every change date should be a week boundary (Friday or last trading day of week)
    for dt in change_dates:
        pos = prices.index.get_loc(dt)
        if pos > 0:
            prev = prices.index[pos - 1]
            # Either this day is a different week than previous, or it's start of data
            assert dt.isocalendar()[1] != prev.isocalendar()[1] or \
                   dt.isocalendar()[0] != prev.isocalendar()[0], \
                f"Signal changed mid-week on {dt.date()}"
