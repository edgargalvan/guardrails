"""Shared test fixtures for the guardrails test suite.

Provides synthetic and real data fixtures used across test modules.
Session-scoped fixtures avoid repeated computation.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src import backtest, filters, metrics, weights

CACHE_PATH = os.path.join(PROJECT_ROOT, "data", "prices.parquet")
THREE_FUND = ["SPY", "TLT", "GLD"]


# ===================================================================
# Synthetic data fixtures
# ===================================================================

@pytest.fixture(scope="session")
def synthetic_rising_prices():
    """500 trading days of linearly rising prices (always above 200dma
    after warmup).

    SPY: 300 → 500, TLT: 100 → 150, GLD: 150 → 250.
    """
    n_days = 500
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame({
        "SPY": np.linspace(300, 500, n_days),
        "TLT": np.linspace(100, 150, n_days),
        "GLD": np.linspace(150, 250, n_days),
    }, index=dates)


@pytest.fixture(scope="session")
def synthetic_falling_prices():
    """500 trading days of linearly falling prices (always below 200dma
    after warmup).

    SPY: 500 → 200, TLT: 150 → 60, GLD: 250 → 100.
    """
    n_days = 500
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame({
        "SPY": np.linspace(500, 200, n_days),
        "TLT": np.linspace(150, 60, n_days),
        "GLD": np.linspace(250, 100, n_days),
    }, index=dates)


# ===================================================================
# Real data fixtures (require data/prices.parquet)
# ===================================================================

has_cache = os.path.exists(CACHE_PATH)
requires_cache = pytest.mark.skipif(
    not has_cache, reason="data/prices.parquet not found"
)


@pytest.fixture(scope="session")
def real_prices():
    """Load cached real market prices. Skips if cache unavailable."""
    if not has_cache:
        pytest.skip("data/prices.parquet not found")
    prices = pd.read_parquet(CACHE_PATH)
    return prices


@pytest.fixture(scope="session")
def strategy_prices(real_prices):
    """SPY/TLT/GLD slice of real prices, NaN-free."""
    return real_prices[THREE_FUND].dropna()


@pytest.fixture(scope="session")
def ew_cash_run(strategy_prices, real_prices):
    """Run EW-cash strategy end-to-end on real data.

    Returns:
        Tuple of (equity_curve, returns, weights_df, in_mask).
    """
    prices = strategy_prices
    in_mask = filters.sma(prices, window=200, frequency="monthly")
    w = weights.equal(prices, in_mask)

    cash_prices = None
    if "SHY" in real_prices.columns:
        cash_prices = real_prices["SHY"]

    equity = backtest.run_backtest(
        prices=prices,
        weights=w,
        exit_mode="cash",
        cash_prices=cash_prices,
        initial_capital=200_000,
        slippage_bps=2.0,
    )

    returns = equity.pct_change().iloc[1:]
    return equity, returns, w, in_mask


@pytest.fixture(scope="session")
def sample_config():
    """Minimal EW-cash strategy config dict."""
    return {
        "strategies": {
            "EW-cash": {
                "universe": THREE_FUND,
                "filter": {"type": "sma", "window": 200, "frequency": "monthly"},
                "weights": {"type": "equal"},
                "exit_mode": "cash",
            },
        },
        "benchmarks": {
            "SPY-BH": {
                "tickers": ["SPY"],
                "weights": {"SPY": 1.0},
            },
        },
        "settings": {
            "initial_capital": 200_000,
            "cash_vehicle": "SHY",
            "costs": {"slippage_bps": 2.0, "commission_per_trade": 0.0},
            "metrics": ["sharpe", "cagr", "max_drawdown"],
            "windows": [
                {"label": "Full Period", "start": "2007-07-01"},
            ],
        },
    }
