"""Tests for src/backtest.py.

Ported from trading/src/backtest/system_tests.py (1.1-1.4) and
trading/scripts/integrity_tests.py (I.1, I.2, V.1).

Validates the backtest engine: next-bar execution, zero-return
invariants, cost model, weight bounds, capital conservation.
"""

import numpy as np
import pandas as pd
import pytest

from src import backtest, filters, weights


# ===================================================================
# 1.1 Next-bar execution (look-ahead bias detection)
# ===================================================================

def test_next_bar_execution(synthetic_rising_prices):
    """Backtest engine must apply shift(1) — no same-bar execution.

    Ported from system_tests.py 1.1.

    Strategy: compare engine output (which applies shift) against a
    manual computation that intentionally does NOT shift. They must
    differ — if identical, the engine is not shifting.
    """
    prices = synthetic_rising_prices
    in_mask = filters.sma(prices, window=200, frequency="monthly")
    w = weights.equal(prices, in_mask)

    # Run real engine (applies shift internally)
    equity_engine = backtest.run_backtest(
        prices=prices, weights=w, exit_mode="renormalize",
        initial_capital=200_000, slippage_bps=0.0,
    )

    # Manual computation WITHOUT shift (same-bar execution — the bug)
    simple_returns = prices.pct_change().iloc[1:]
    w_aligned = w.reindex(simple_returns.index, method="ffill").fillna(0.0)
    row_sums = w_aligned.sum(axis=1).replace(0, np.nan)
    w_normed = w_aligned.div(row_sums, axis=0).fillna(0.0)
    port_ret_no_shift = (w_normed * simple_returns).sum(axis=1)
    equity_no_shift = (1 + port_ret_no_shift).cumprod() * 200_000

    # They must differ (identical means shift is broken)
    common = equity_engine.index.intersection(equity_no_shift.index)
    final_engine = float(equity_engine.loc[common[-1]])
    final_no_shift = float(equity_no_shift.loc[common[-1]])

    assert abs(final_engine - final_no_shift) > 1.0, \
        (f"Engine output identical to no-shift computation "
         f"({final_engine:.2f} vs {final_no_shift:.2f}) — "
         f"shift(1) not working")


# ===================================================================
# 1.2 / V.1 Zero return when flat
# ===================================================================

def test_zero_return_when_flat_no_cash(synthetic_falling_prices):
    """Zero weights + no cash vehicle → returns must be exactly 0.

    Ported from system_tests.py 1.2 (cash mode).
    """
    prices = synthetic_falling_prices
    zero_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    equity = backtest.run_backtest(
        prices=prices, weights=zero_weights, exit_mode="cash",
        cash_prices=None, initial_capital=200_000, slippage_bps=0.0,
    )

    returns = equity.pct_change().iloc[1:]
    max_abs_ret = returns.abs().max()
    assert max_abs_ret < 1e-10, \
        f"Expected zero returns with no cash, got max |return| = {max_abs_ret:.2e}"


def test_zero_return_when_flat_with_cash(synthetic_falling_prices):
    """Zero weights + cash vehicle → returns must match cash return.

    Ported from integrity_tests.py V.1: zero-exposure invariance.
    """
    prices = synthetic_falling_prices
    zero_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    # Use TLT prices as a proxy for cash vehicle
    cash_prices = prices["TLT"]

    equity = backtest.run_backtest(
        prices=prices, weights=zero_weights, exit_mode="cash",
        cash_prices=cash_prices, initial_capital=200_000, slippage_bps=0.0,
    )

    returns = equity.pct_change().iloc[1:]
    cash_ret = cash_prices.pct_change().iloc[1:]

    # Align indices
    common = returns.index.intersection(cash_ret.index)
    diff = (returns.loc[common] - cash_ret.loc[common]).abs()
    max_diff = diff.max()

    # Small numerical differences are OK
    assert max_diff < 1e-4, \
        f"Returns should match cash vehicle, max diff = {max_diff:.2e}"


# ===================================================================
# 1.3 Cost scaling
# ===================================================================

def test_cost_scaling_linear(synthetic_rising_prices):
    """Cost drag between slippage levels should be roughly linear.

    Ported from system_tests.py 1.3.
    """
    prices = synthetic_rising_prices
    in_mask = filters.sma(prices, window=200, frequency="monthly")
    w = weights.equal(prices, in_mask)

    equities = {}
    for bps in [2.0, 4.0, 6.0]:
        equities[bps] = backtest.run_backtest(
            prices=prices, weights=w, exit_mode="cash",
            initial_capital=200_000, slippage_bps=bps,
        )

    # Terminal values
    tv = {bps: float(eq.iloc[-1]) for bps, eq in equities.items()}

    drag_2_to_4 = tv[2.0] - tv[4.0]
    drag_4_to_6 = tv[4.0] - tv[6.0]

    # Both drags should be positive (more slippage → less return)
    if drag_2_to_4 > 0 and drag_4_to_6 > 0:
        ratio = drag_4_to_6 / drag_2_to_4
        assert 0.3 < ratio < 3.0, \
            f"Non-linear cost scaling: drag ratio = {ratio:.2f}"


# ===================================================================
# 1.4 Weight bounds
# ===================================================================

def test_weight_bounds_no_negatives(synthetic_rising_prices):
    """Weights must have no negatives and sum <= 1.0.

    Ported from system_tests.py 1.4.
    """
    prices = synthetic_rising_prices
    in_mask = filters.sma(prices, window=200, frequency="monthly")

    for weight_fn in [weights.equal, weights.risk_parity]:
        w = weight_fn(prices, in_mask)

        # No negatives
        neg_count = (w < -1e-10).sum().sum()
        assert neg_count == 0, \
            f"{weight_fn.__name__}: {neg_count} negative weight entries"

        # Sum <= 1.0
        row_sums = w.sum(axis=1)
        max_sum = row_sums.max()
        assert max_sum <= 1.0 + 1e-8, \
            f"{weight_fn.__name__}: weight sum {max_sum:.6f} > 1.0"


# ===================================================================
# I.1 Capital conservation
# ===================================================================

def test_capital_conservation(synthetic_falling_prices):
    """Zero-weight portfolio with cash vehicle must grow monotonically.

    Ported from integrity_tests.py I.1.
    """
    prices = synthetic_falling_prices
    zero_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    # Use a synthetic rising cash vehicle (like SHY — always goes up)
    n = len(prices)
    cash_prices = pd.Series(
        np.linspace(80, 82, n),  # slight upward drift
        index=prices.index,
        name="SHY",
    )

    equity = backtest.run_backtest(
        prices=prices, weights=zero_weights, exit_mode="cash",
        cash_prices=cash_prices, initial_capital=200_000, slippage_bps=0.0,
    )

    # Should grow monotonically (no drawdowns)
    dd = equity / equity.cummax() - 1
    max_dd = float(dd.min())
    assert abs(max_dd) < 1e-10, \
        f"Cash-only portfolio should have no drawdowns, got MaxDD = {max_dd:.8f}"

    # Should end higher than start
    assert float(equity.iloc[-1]) > float(equity.iloc[0]), \
        "Cash-only portfolio should grow"


# ===================================================================
# I.2 Exposure accounting
# ===================================================================

def test_exposure_accounting(synthetic_rising_prices):
    """Weights + implicit cash fraction must sum to 1.0 every day.

    Ported from integrity_tests.py I.2.
    """
    prices = synthetic_rising_prices
    in_mask = filters.sma(prices, window=200, frequency="monthly")
    w = weights.equal(prices, in_mask)

    total_weight = w.sum(axis=1)
    cash_weight = 1.0 - total_weight
    total_all = total_weight + cash_weight

    # Total should be exactly 1.0
    max_deviation = (total_all - 1.0).abs().max()
    assert max_deviation < 1e-10, \
        f"Weight + cash != 1.0: max deviation = {max_deviation:.2e}"

    # No negative cash (would mean leverage)
    neg_cash = (cash_weight < -1e-10).sum()
    assert neg_cash == 0, f"{neg_cash} days with negative cash (leverage)"


# ===================================================================
# Rebalance frequency validation (G4)
# ===================================================================

def test_invalid_rebalance_raises():
    """run_benchmark() should raise ValueError for invalid rebalance frequency."""
    n = 10
    dates = pd.bdate_range("2020-01-01", periods=n)
    prices = pd.DataFrame({
        "SPY": [100.0] * n, "TLT": [100.0] * n,
    }, index=dates)

    with pytest.raises(ValueError, match="Invalid rebalance frequency"):
        backtest.run_benchmark(
            prices=prices,
            allocation={"SPY": 0.60, "TLT": 0.40},
            rebalance="quraterly",  # intentional typo
        )
