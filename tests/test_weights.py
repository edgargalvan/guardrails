"""Tests for src/weights.py.

Validates weight function invariants: sum <= 1.0, no negatives,
zero for filtered-out assets, correct values for known inputs.
"""

import numpy as np
import pandas as pd
import pytest

from src import filters, weights


# ===================================================================
# Helpers
# ===================================================================

@pytest.fixture(scope="module")
def prices_and_mask(synthetic_rising_prices):
    """Compute prices and in_mask for weight tests."""
    prices = synthetic_rising_prices
    in_mask = filters.sma(prices, window=200, frequency="monthly")
    return prices, in_mask


@pytest.fixture(scope="module")
def mixed_mask(synthetic_rising_prices):
    """Create a mask where some assets are in and some are out."""
    prices = synthetic_rising_prices
    n = len(prices)
    mask = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    # Force TLT and GLD below trend for a stretch
    mask.iloc[300:400, 1] = False  # TLT out
    mask.iloc[350:450, 2] = False  # GLD out
    return prices, mask


# ===================================================================
# Equal weight tests
# ===================================================================

def test_equal_value_is_one_over_n_total(prices_and_mask):
    """Each 'in' asset gets exactly 1/n_total."""
    prices, in_mask = prices_and_mask
    w = weights.equal(prices, in_mask)
    n_total = len(prices.columns)

    # Where in_mask is True, weight should be 1/n_total
    for col in prices.columns:
        invested = in_mask[col]
        w_invested = w.loc[invested, col]
        if len(w_invested) > 0:
            np.testing.assert_allclose(
                w_invested.values, 1.0 / n_total,
                err_msg=f"{col}: expected {1.0/n_total}, got {w_invested.iloc[0]}"
            )


def test_equal_zero_for_filtered_out(mixed_mask):
    """Filtered-out assets must have weight exactly 0."""
    prices, in_mask = mixed_mask
    w = weights.equal(prices, in_mask)

    for col in prices.columns:
        filtered_out = ~in_mask[col]
        w_out = w.loc[filtered_out, col]
        if len(w_out) > 0:
            assert (w_out == 0.0).all(), \
                f"{col}: non-zero weight on {(w_out != 0).sum()} filtered-out days"


def test_equal_sum_leq_one(mixed_mask):
    """Equal weights must sum to <= 1.0 on every day."""
    prices, in_mask = mixed_mask
    w = weights.equal(prices, in_mask)
    row_sums = w.sum(axis=1)
    assert (row_sums <= 1.0 + 1e-10).all(), \
        f"Weight sum exceeds 1.0: max = {row_sums.max():.6f}"


# ===================================================================
# Momentum weight tests
# ===================================================================

def test_momentum_sum_leq_one(prices_and_mask):
    """Momentum weights must sum to <= 1.0 on every day."""
    prices, in_mask = prices_and_mask
    w = weights.momentum(prices, in_mask)
    row_sums = w.sum(axis=1)
    assert (row_sums <= 1.0 + 1e-10).all(), \
        f"Weight sum exceeds 1.0: max = {row_sums.max():.6f}"


def test_momentum_zero_for_filtered_out(mixed_mask):
    """Filtered-out assets must have momentum weight = 0."""
    prices, in_mask = mixed_mask
    w = weights.momentum(prices, in_mask)

    for col in prices.columns:
        filtered_out = ~in_mask[col]
        w_out = w.loc[filtered_out, col]
        if len(w_out) > 0:
            assert (w_out == 0.0).all(), \
                f"{col}: non-zero momentum weight on filtered-out days"


def test_momentum_ranking_correct():
    """Asset with clearly dominant momentum should get the highest weight."""
    n_days = 400
    dates = pd.bdate_range("2020-01-01", periods=n_days)

    # SPY strongly rising, TLT flat, GLD slightly down
    prices = pd.DataFrame({
        "SPY": np.linspace(300, 600, n_days),   # strong momentum
        "TLT": np.full(n_days, 100.0),           # flat
        "GLD": np.linspace(200, 180, n_days),    # negative
    }, index=dates)

    in_mask = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    w = weights.momentum(prices, in_mask, split=[0.70, 0.20, 0.10])

    # After momentum warmup (~252 days), SPY should have highest weight
    post_warmup = w.iloc[300:]
    if len(post_warmup) > 0:
        spy_avg = post_warmup["SPY"].mean()
        tlt_avg = post_warmup["TLT"].mean()
        gld_avg = post_warmup["GLD"].mean()
        assert spy_avg > tlt_avg, \
            f"SPY weight ({spy_avg:.3f}) should exceed TLT ({tlt_avg:.3f})"
        assert spy_avg > gld_avg, \
            f"SPY weight ({spy_avg:.3f}) should exceed GLD ({gld_avg:.3f})"


# ===================================================================
# Fixed weight tests
# ===================================================================

def test_fixed_matches_config(mixed_mask):
    """Fixed weights must match allocation dict × in_mask."""
    prices, in_mask = mixed_mask
    allocation = {"SPY": 0.70, "TLT": 0.20, "GLD": 0.10}
    w = weights.fixed(prices, in_mask, allocation=allocation)

    for col, target in allocation.items():
        invested = in_mask[col]
        # Where invested, weight should be target
        w_in = w.loc[invested, col]
        if len(w_in) > 0:
            np.testing.assert_allclose(w_in.values, target, atol=1e-10)
        # Where filtered out, weight should be 0
        w_out = w.loc[~invested, col]
        if len(w_out) > 0:
            assert (w_out == 0.0).all()


# ===================================================================
# Risk parity weight tests
# ===================================================================

def test_risk_parity_sum_leq_one(prices_and_mask):
    """Risk parity weights must sum to <= 1.0 on every day."""
    prices, in_mask = prices_and_mask
    w = weights.risk_parity(prices, in_mask)
    row_sums = w.sum(axis=1)
    assert (row_sums <= 1.0 + 1e-10).all(), \
        f"RP weight sum exceeds 1.0: max = {row_sums.max():.6f}"


# ===================================================================
# Edge case tests (G1)
# ===================================================================

def test_fixed_missing_ticker_in_allocation():
    """fixed() with allocation missing a ticker should give that ticker 0 weight."""
    n = 10
    dates = pd.bdate_range("2020-01-01", periods=n)
    prices = pd.DataFrame({
        "SPY": [100.0] * n, "TLT": [100.0] * n, "GLD": [100.0] * n,
    }, index=dates)
    in_mask = pd.DataFrame(True, index=dates, columns=prices.columns)

    # Allocation only mentions SPY and TLT, not GLD
    w = weights.fixed(prices, in_mask, allocation={"SPY": 0.60, "TLT": 0.40})

    assert (w["GLD"] == 0.0).all(), "Ticker not in allocation should have 0 weight"
    np.testing.assert_allclose(w["SPY"].values, 0.60, atol=1e-10)
    np.testing.assert_allclose(w["TLT"].values, 0.40, atol=1e-10)


def test_fixed_allocation_over_one():
    """fixed() with allocation > 1.0 should still produce those weights.

    The backtest engine (not the weight function) handles leverage detection.
    """
    n = 10
    dates = pd.bdate_range("2020-01-01", periods=n)
    prices = pd.DataFrame({
        "SPY": [100.0] * n, "TLT": [100.0] * n,
    }, index=dates)
    in_mask = pd.DataFrame(True, index=dates, columns=prices.columns)

    w = weights.fixed(prices, in_mask, allocation={"SPY": 0.80, "TLT": 0.50})

    # Weights should be exactly what was specified (1.3 total)
    assert abs(w.sum(axis=1).iloc[0] - 1.3) < 1e-10, \
        f"Sum should be 1.3, got {w.sum(axis=1).iloc[0]}"


def test_momentum_identical_scores():
    """All assets with identical momentum should get equal rank weights.

    When all scores are the same, sort_values returns them in arbitrary
    order, but the weight distribution should be deterministic.
    """
    n = 253
    dates = pd.bdate_range("2020-01-01", periods=n)
    # All three assets have identical price paths → identical momentum
    prices = pd.DataFrame({
        "A": np.linspace(100, 120, n),
        "B": np.linspace(100, 120, n),
        "C": np.linspace(100, 120, n),
    }, index=dates)
    in_mask = pd.DataFrame(True, index=dates, columns=prices.columns)

    w = weights.momentum(prices, in_mask, split=[0.70, 0.20, 0.10])
    last = dates[-1]

    # All weights should sum to 1.0 (all assets in)
    assert abs(w.loc[last].sum() - 1.0) < 0.01, \
        f"Total weight should be ~1.0, got {w.loc[last].sum()}"
    # All three weights should be positive
    for col in ["A", "B", "C"]:
        assert w.loc[last, col] > 0, f"{col} should have positive weight"
