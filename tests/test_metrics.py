"""Tests for src/metrics.py.

Validates metric functions with known inputs and edge cases.
"""

import numpy as np
import pandas as pd
import pytest

from src import metrics


# ===================================================================
# Sharpe ratio tests
# ===================================================================

def test_sharpe_positive_for_uptrend():
    """Positive-mean returns with noise should produce positive Sharpe."""
    n = 500
    dates = pd.bdate_range("2020-01-01", periods=n)
    # Positive mean + realistic daily noise
    np.random.seed(42)
    returns = pd.Series(0.0003 + np.random.randn(n) * 0.01, index=dates)
    result = metrics.sharpe(returns)
    assert result > 0, f"Sharpe should be positive for positive-mean returns, got {result}"


def test_sharpe_returns_zero_for_zero_std():
    """Zero returns (std=0) → NaN Sharpe → _safe() returns 0.0."""
    n = 500
    dates = pd.bdate_range("2020-01-01", periods=n)
    returns = pd.Series(0.0, index=dates)
    result = metrics.sharpe(returns)
    assert result == 0.0, f"Sharpe should be 0 for flat returns, got {result}"


# ===================================================================
# Max drawdown tests
# ===================================================================

def test_max_drawdown_zero_for_monotonic():
    """Monotonically increasing equity should have MaxDD = 0."""
    n = 500
    dates = pd.bdate_range("2020-01-01", periods=n)
    equity = pd.Series(np.linspace(200_000, 400_000, n), index=dates)
    result = metrics.max_drawdown(equity)
    assert result == 0.0, f"MaxDD should be 0 for monotonic equity, got {result}"


def test_max_drawdown_negative():
    """Any equity with a dip should have MaxDD < 0."""
    n = 500
    dates = pd.bdate_range("2020-01-01", periods=n)
    values = np.linspace(200_000, 400_000, n)
    values[200:250] = 180_000  # dip
    equity = pd.Series(values, index=dates)
    result = metrics.max_drawdown(equity)
    assert result < 0, f"MaxDD should be negative for equity with a dip, got {result}"


# ===================================================================
# CAGR tests
# ===================================================================

def test_cagr_known_values():
    """CAGR for 200K → 400K in 10 years ≈ 7.18%."""
    n = 252 * 10  # 10 years
    dates = pd.bdate_range("2010-01-01", periods=n)
    equity = pd.Series(np.linspace(200_000, 400_000, n), index=dates)
    # Override start/end for exact calculation
    equity.iloc[0] = 200_000
    equity.iloc[-1] = 400_000
    result = metrics.cagr(equity)
    expected = 2.0 ** (1 / 10) - 1  # ≈ 0.07177
    assert abs(result - expected) < 0.001, \
        f"CAGR should be ≈{expected:.4f}, got {result:.4f}"


# ===================================================================
# compute_all tests
# ===================================================================

def test_compute_all_returns_all_keys():
    """compute_all() must return all requested metric names."""
    n = 500
    dates = pd.bdate_range("2020-01-01", periods=n)
    returns = pd.Series(np.random.randn(n) * 0.01, index=dates)
    equity = (1 + returns).cumprod() * 200_000

    requested = ["sharpe", "cagr", "max_drawdown", "sortino", "annual_vol"]
    result = metrics.compute_all(returns, equity, requested)

    for name in requested:
        assert name in result, f"Missing metric: {name}"
        assert result[name] is not None, f"Metric {name} is None"


def test_compute_all_full_config_metrics():
    """compute_all() must handle all 11 metrics from default config (G3)."""
    np.random.seed(42)
    n = 500
    dates = pd.bdate_range("2020-01-01", periods=n)
    returns = pd.Series(0.0003 + np.random.randn(n) * 0.01, index=dates)
    equity = (1 + returns).cumprod() * 200_000
    benchmark = pd.Series(0.0002 + np.random.randn(n) * 0.012, index=dates)

    # All 11 metrics from configs/default.yaml
    all_metrics = [
        "sharpe", "sortino", "cagr", "annual_vol", "max_drawdown",
        "calmar", "ulcer_index", "worst_month", "worst_year",
        "upside_capture", "downside_capture",
    ]
    result = metrics.compute_all(returns, equity, all_metrics, benchmark)

    for name in all_metrics:
        assert name in result, f"Missing metric: {name}"
        assert result[name] is not None, f"Metric {name} is None"
        assert isinstance(result[name], float), f"Metric {name} should be float, got {type(result[name])}"


def test_max_drawdown_all_nan_returns_zero():
    """All-NaN equity curve should return 0.0, not NaN (M5 fix)."""
    n = 10
    dates = pd.bdate_range("2020-01-01", periods=n)
    equity = pd.Series([float("nan")] * n, index=dates)
    result = metrics.max_drawdown(equity)
    assert result == 0.0, f"All-NaN equity should return 0.0, got {result}"


def test_safe_handles_non_numeric():
    """_safe() should return 0.0 for non-numeric types (M3 fix)."""
    from src.metrics import _safe
    assert _safe(None) == 0.0
    assert _safe(float("nan")) == 0.0
    assert _safe(float("inf")) == 0.0
    assert _safe("not_a_number") == 0.0
    assert _safe([1, 2, 3]) == 0.0
    assert _safe(42.0) == 42.0
