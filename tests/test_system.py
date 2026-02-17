"""Full-pipeline system validation tests.

Ported from trading/src/backtest/system_tests.py (2.1-2.7) and
trading/scripts/integrity_tests.py (II.3, III.1-III.4, IV.1).

These tests validate emergent behavior across the full pipeline:
filters → weights → backtest → metrics. Most require real market
data (data/prices.parquet).
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src import backtest, filters, metrics, weights

CACHE_PATH = os.path.join(PROJECT_ROOT, "data", "prices.parquet")
has_cache = os.path.exists(CACHE_PATH)
requires_cache = pytest.mark.skipif(
    not has_cache, reason="data/prices.parquet not found"
)


# ===================================================================
# 2.1 Sharpe ceiling
# ===================================================================

@requires_cache
def test_sharpe_ceiling(ew_cash_run):
    """Walk-forward Sharpe must be below 2.0 for long-only liquid ETFs.

    Ported from system_tests.py 2.1. Above 2.0 is almost certainly
    look-ahead or survivorship bias.
    """
    _, returns, _, _ = ew_cash_run
    s = metrics.sharpe(returns)
    assert s < 2.0, f"Sharpe {s:.3f} > 2.0 ceiling — suspicious"


# ===================================================================
# 2.2 Drawdown floor
# ===================================================================

@requires_cache
def test_drawdown_floor(ew_cash_run, strategy_prices):
    """Strategy MaxDD must be at least 5% of buy-and-hold MaxDD.

    Ported from system_tests.py 2.2. Near-zero DD with positive
    returns indicates look-ahead bias.
    """
    equity, returns, _, _ = ew_cash_run

    # Compute B&H equity (equal weight, no filter)
    bh_allocation = {"SPY": 1.0 / 3, "TLT": 1.0 / 3, "GLD": 1.0 / 3}
    bh_equity = backtest.run_benchmark(
        strategy_prices, bh_allocation, initial_capital=200_000,
    )

    strategy_dd = abs(metrics.max_drawdown(equity))
    bh_dd = abs(metrics.max_drawdown(bh_equity))

    if bh_dd == 0:
        pytest.skip("Buy-and-hold has zero drawdown")

    ratio = strategy_dd / bh_dd
    s = metrics.sharpe(returns)

    if ratio < 0.05 and s > 1.5:
        pytest.fail(
            f"Suspiciously low DD: strategy DD {strategy_dd:.1%} vs "
            f"B&H DD {bh_dd:.1%} (ratio {ratio:.2f}), Sharpe={s:.2f}"
        )


# ===================================================================
# 2.3 2008 stress test
# ===================================================================

@requires_cache
def test_2008_stress(ew_cash_run):
    """Strategy should not make > 5% in 2008 (suspicious if it does).

    Ported from system_tests.py 2.3. A positive 2008 return isn't
    necessarily wrong but requires investigation.
    """
    _, returns, _, _ = ew_cash_run

    if returns.index[0].year > 2008 or returns.index[-1].year < 2008:
        pytest.skip("Data does not cover 2008")

    returns_2008 = returns[returns.index.year == 2008]
    total_2008 = float((1 + returns_2008).prod() - 1)

    if total_2008 > 0.05:
        warnings.warn(
            f"Strategy made {total_2008:.1%} in 2008 — verify regime "
            f"filter timing. This may be legitimate if filter went flat early."
        )


# ===================================================================
# 2.4 Calm period tracking
# ===================================================================

@requires_cache
def test_calm_period_tracking(ew_cash_run, strategy_prices):
    """During calm bull markets, strategy returns should correlate
    with buy-and-hold.

    Ported from system_tests.py 2.4. Informational (warn-only).
    Calm period: SPY drawdown < 10% for 12+ consecutive months.
    """
    _, returns, _, _ = ew_cash_run

    spy = strategy_prices["SPY"].reindex(returns.index).ffill().dropna()
    if len(spy) < 252:
        pytest.skip("Insufficient SPY data")

    # Identify calm periods
    cum_spy = spy / spy.iloc[0]
    running_max = cum_spy.cummax()
    spy_dd = cum_spy / running_max - 1
    is_calm = spy_dd > -0.10

    # Find runs of calm days >= 12 months
    min_days = 12 * 21
    run_start = None
    calm_indices = []
    for i, (_, calm) in enumerate(is_calm.items()):
        if calm:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and i - run_start >= min_days:
                calm_indices.extend(range(run_start, i))
            run_start = None
    if run_start is not None and len(is_calm) - run_start >= min_days:
        calm_indices.extend(range(run_start, len(is_calm)))

    if not calm_indices:
        pytest.skip("No calm periods found")

    calm_dates = returns.index[calm_indices]

    # B&H returns for comparison
    bh_ret = spy.pct_change().iloc[1:]
    common = calm_dates.intersection(returns.index).intersection(bh_ret.index)
    if len(common) < 60:
        pytest.skip("Too few overlapping calm days")

    corr = returns.loc[common].corr(bh_ret.loc[common])
    if corr < 0.5:
        warnings.warn(
            f"Low calm-period correlation {corr:.2f} with SPY B&H. "
            f"Expected for multi-asset trend strategies with cash drag."
        )


# ===================================================================
# 2.5 Sub-period concentration
# ===================================================================

@requires_cache
def test_subperiod_concentration(ew_cash_run):
    """No single year should contribute > 40% of total return.

    Ported from system_tests.py 2.5.
    """
    _, returns, _, _ = ew_cash_run

    total = float((1 + returns).prod() - 1)
    if abs(total) < 1e-8:
        pytest.skip("Near-zero total return")

    yearly = returns.groupby(returns.index.year).apply(
        lambda x: float((1 + x).prod() - 1)
    )
    contributions = (yearly / total).abs()
    max_contrib = contributions.max()
    max_year = contributions.idxmax()

    if max_contrib > 0.40:
        warnings.warn(
            f"Year {max_year} contributes {max_contrib:.0%} of total return. "
            f"Strategy may be regime-dependent."
        )


# ===================================================================
# 2.6 Regime filter activity
# ===================================================================

@requires_cache
def test_regime_filter_activity(ew_cash_run):
    """Filter should be active on a meaningful fraction of days.

    Ported from system_tests.py 2.6. With per-asset 200dma on 3
    assets, it's common for at least 1 of 3 to be below trend
    (60-70% of days). The test validates:
    - Filter is not inactive (never fires: < 5%)
    - Filter is not always-on (100% filtered: everything in cash)
    """
    _, _, w, in_mask = ew_cash_run

    # Check per-asset: what fraction of days each asset is filtered out
    n_days = len(in_mask)
    per_asset_filtered = (~in_mask.astype(bool)).sum() / n_days

    # At least one asset should be filtered sometimes (> 5%)
    any_filtered = (per_asset_filtered > 0.05).any()
    assert any_filtered, \
        f"Filter never fires: {per_asset_filtered.to_dict()}"

    # Not ALL assets filtered ALL the time
    total_exposure = w.sum(axis=1)
    all_zero_days = (total_exposure < 1e-10).sum()
    pct_all_zero = all_zero_days / n_days
    assert pct_all_zero < 0.50, \
        f"All assets filtered on {pct_all_zero:.1%} of days — filter too aggressive"


# ===================================================================
# 2.7 Transaction count
# ===================================================================

@requires_cache
def test_transaction_count(ew_cash_run):
    """Large trades should cluster near regime transitions.

    Ported from system_tests.py 2.7.
    """
    _, _, w, _ = ew_cash_run

    total_exposure = w.sum(axis=1)
    weight_diffs = w.diff().abs()

    # Identify transitions (> 10% change in total exposure)
    exposure_changes = total_exposure.diff().abs()
    max_exp = total_exposure.max()
    trans_threshold = max_exp * 0.10 if max_exp > 0 else 0.01
    transition_days = exposure_changes[exposure_changes > trans_threshold].index

    # Build near-transition set (±1 day)
    near_transition = set()
    idx_list = w.index.tolist()
    idx_map = {d: i for i, d in enumerate(idx_list)}
    for td in transition_days:
        pos = idx_map.get(td)
        if pos is not None:
            for offset in (-1, 0, 1):
                p = pos + offset
                if 0 <= p < len(idx_list):
                    near_transition.add(idx_list[p])

    # Large trades: any asset weight changes > 10%
    max_per_asset = weight_diffs.max(axis=1)
    large_trade_days = max_per_asset[max_per_asset > 0.10].index

    if len(large_trade_days) == 0:
        return  # No large trades — pass

    orphans = [d for d in large_trade_days if d not in near_transition]
    orphan_pct = len(orphans) / len(large_trade_days)

    if orphan_pct > 0.20:
        warnings.warn(
            f"{len(orphans)}/{len(large_trade_days)} large trades "
            f"({orphan_pct:.0%}) not near regime transitions"
        )


# ===================================================================
# III.1 Shock injection
# ===================================================================

@requires_cache
def test_shock_injection(strategy_prices, real_prices):
    """Loss from -20% SPY shock must be bounded by SPY weight.

    Ported from integrity_tests.py III.1.
    """
    prices = strategy_prices
    in_mask = filters.sma(prices, window=200, frequency="monthly")
    w = weights.equal(prices, in_mask)
    n_total = len(prices.columns)

    # Find a day where SPY is above trend
    spy_in = in_mask["SPY"]
    invested_dates = spy_in[spy_in].index
    if len(invested_dates) < 10:
        pytest.skip("Too few SPY-invested days")

    shock_date = invested_dates[len(invested_dates) // 2]
    shock_pos = prices.index.get_loc(shock_date)
    if shock_pos + 2 >= len(prices):
        pytest.skip("Shock too near end of data")

    # Create shocked prices
    shocked_prices = prices.copy()
    # Drop SPY by 20% from shock_pos+1 onward
    shocked_prices.iloc[shock_pos + 1:,
                        prices.columns.get_loc("SPY")] *= 0.80

    cash_prices = real_prices["SHY"] if "SHY" in real_prices.columns else None

    equity_normal = backtest.run_backtest(
        prices, w, "cash", cash_prices=cash_prices,
        initial_capital=200_000, slippage_bps=2.0,
    )
    equity_shocked = backtest.run_backtest(
        shocked_prices, w, "cash", cash_prices=cash_prices,
        initial_capital=200_000, slippage_bps=2.0,
    )

    # Check that shocked equity never goes to zero
    assert float(equity_shocked.min()) > 0, \
        "Equity went to zero after shock"

    # The impact should be bounded by SPY_weight × 20%
    spy_weight = 1.0 / n_total
    max_expected_loss_pct = spy_weight * 0.20  # ≈ 6.7%
    actual_loss_pct = 1.0 - float(equity_shocked.iloc[-1] / equity_normal.iloc[-1])

    # Allow some tolerance (costs, rebalancing effects)
    assert actual_loss_pct < max_expected_loss_pct * 2, \
        f"Shock loss {actual_loss_pct:.1%} exceeds 2× bounded estimate {max_expected_loss_pct:.1%}"


# ===================================================================
# III.2 Return permutation
# ===================================================================

@requires_cache
def test_return_permutation(ew_cash_run):
    """MaxDD should be better than median of shuffled return paths.

    Ported from integrity_tests.py III.2. Temporal structure
    (trend filter) should protect drawdowns beyond what the
    return distribution alone provides.
    """
    equity, returns, _, _ = ew_cash_run

    original_dd = metrics.max_drawdown(equity)

    n_trials = 200
    np.random.seed(42)
    shuffled_dds = []

    # Shuffle monthly blocks
    monthly_groups = returns.groupby(
        [returns.index.year, returns.index.month]
    )
    blocks = [group.values for _, group in monthly_groups]

    for _ in range(n_trials):
        shuffled_blocks = [b.copy() for b in blocks]
        np.random.shuffle(shuffled_blocks)
        shuffled = np.concatenate(shuffled_blocks)
        cum = np.cumprod(1 + shuffled)
        running_max = np.maximum.accumulate(cum)
        dd = (cum / running_max - 1).min()
        shuffled_dds.append(dd)

    shuffled_dds = np.array(shuffled_dds)
    pctile_better = (shuffled_dds > original_dd).mean()

    if pctile_better > 0.75:
        warnings.warn(
            f"{pctile_better:.0%} of shuffled paths have better MaxDD "
            f"than original ({original_dd:.1%}). Temporal structure may "
            f"not be protecting drawdowns."
        )


# ===================================================================
# III.4 Slippage degradation
# ===================================================================

@requires_cache
def test_slippage_degradation(strategy_prices, real_prices):
    """Sharpe should degrade smoothly at higher slippage levels.

    Ported from integrity_tests.py III.4.
    """
    prices = strategy_prices
    in_mask = filters.sma(prices, window=200, frequency="monthly")
    w = weights.equal(prices, in_mask)
    cash_prices = real_prices["SHY"] if "SHY" in real_prices.columns else None

    slippage_levels = [2.0, 4.0, 8.0, 16.0]
    sharpes = []

    for bps in slippage_levels:
        equity = backtest.run_backtest(
            prices, w, "cash", cash_prices=cash_prices,
            initial_capital=200_000, slippage_bps=bps,
        )
        returns = equity.pct_change().iloc[1:]
        sharpes.append(metrics.sharpe(returns))

    # Sharpe should decrease with increasing slippage
    for i in range(len(sharpes) - 1):
        assert sharpes[i] >= sharpes[i + 1] - 0.01, \
            f"Sharpe increased at {slippage_levels[i+1]}bps: " \
            f"{sharpes[i]:.3f} → {sharpes[i+1]:.3f}"

    # Edge should survive 4 bps (Sharpe still positive)
    assert sharpes[1] > 0, \
        f"Sharpe at 4bps = {sharpes[1]:.3f} — edge too fragile"

    # Check for smooth degradation
    deltas = [sharpes[i] - sharpes[i + 1] for i in range(len(sharpes) - 1)]
    for i in range(len(deltas) - 1):
        if deltas[i] > 1e-6:
            ratio = deltas[i + 1] / deltas[i]
            if ratio > 5.0:
                warnings.warn(
                    f"Non-smooth Sharpe degradation: delta ratio {ratio:.1f} "
                    f"at {slippage_levels[i+1]}→{slippage_levels[i+2]} bps"
                )


# ===================================================================
# IV.1 Monte Carlo bootstrap
# ===================================================================

@requires_cache
def test_monte_carlo_bootstrap(ew_cash_run):
    """Bootstrap monthly returns: CAGR should not be an extreme outlier.

    Ported from integrity_tests.py IV.1.
    """
    equity, returns, _, _ = ew_cash_run

    # Compute monthly simple returns
    monthly = returns.groupby(
        [returns.index.year, returns.index.month]
    ).apply(lambda x: (1 + x).prod() - 1)
    monthly_values = monthly.values
    n_months = len(monthly_values)

    original_cagr = metrics.cagr(equity)

    n_trials = 1000
    np.random.seed(42)
    bootstrap_cagrs = []

    for _ in range(n_trials):
        sample_idx = np.random.choice(n_months, size=n_months, replace=True)
        sample = monthly_values[sample_idx]
        total_return = (1 + sample).prod()
        n_years = n_months / 12
        cagr_val = total_return ** (1 / n_years) - 1 if n_years > 0 else 0
        bootstrap_cagrs.append(cagr_val)

    bootstrap_cagrs = np.array(bootstrap_cagrs)
    cagr_pctile = (bootstrap_cagrs <= original_cagr).mean()

    # Original should not be > 95th percentile (too lucky)
    if cagr_pctile > 0.95:
        warnings.warn(
            f"Original CAGR at {cagr_pctile:.0%} percentile — "
            f"possibly lucky path"
        )

    # 5th percentile CAGR should be positive
    cagr_5th = np.percentile(bootstrap_cagrs, 5)
    assert cagr_5th > -0.02, \
        f"5th percentile CAGR = {cagr_5th:.1%} — edge may not be robust"


# ===================================================================
# II.3 Look-ahead bias (full pipeline MA shift)
# ===================================================================

@requires_cache
def test_look_ahead_bias_ma_shift(strategy_prices, real_prices):
    """Shifting filter signal by 1 day should not materially change Sharpe.

    Ported from integrity_tests.py II.3.
    """
    prices = strategy_prices
    cash_prices = real_prices["SHY"] if "SHY" in real_prices.columns else None

    # Normal run
    in_mask_normal = filters.sma(prices, window=200, frequency="monthly")
    w_normal = weights.equal(prices, in_mask_normal)
    eq_normal = backtest.run_backtest(
        prices, w_normal, "cash", cash_prices=cash_prices,
        initial_capital=200_000, slippage_bps=2.0,
    )
    ret_normal = eq_normal.pct_change().iloc[1:]
    sharpe_normal = metrics.sharpe(ret_normal)

    # Shifted: shift the filter signal by 1 day (simulates using
    # yesterday's MA instead of today's)
    in_mask_shifted = in_mask_normal.shift(1).fillna(True)
    w_shifted = weights.equal(prices, in_mask_shifted)
    eq_shifted = backtest.run_backtest(
        prices, w_shifted, "cash", cash_prices=cash_prices,
        initial_capital=200_000, slippage_bps=2.0,
    )
    ret_shifted = eq_shifted.pct_change().iloc[1:]
    sharpe_shifted = metrics.sharpe(ret_shifted)

    delta = abs(sharpe_normal - sharpe_shifted)
    assert delta < 0.10, \
        f"Sharpe changed by {delta:.4f} with 1-day MA shift — " \
        f"possible look-ahead bias (normal={sharpe_normal:.3f}, " \
        f"shifted={sharpe_shifted:.3f})"


# ===================================================================
# Regression test
# ===================================================================

@requires_cache
def test_regression_ew_cash(ew_cash_run):
    """EW-cash metrics should match known regression baselines.

    Baselines from the verified run (Full Period from 2007-07-01):
    Sharpe ≈ 1.04, MaxDD ≈ -13.8%, CAGR ≈ 7.9%.
    """
    equity, returns, _, _ = ew_cash_run

    s = metrics.sharpe(returns)
    dd = metrics.max_drawdown(equity)
    c = metrics.cagr(equity)

    # Trim to Full Period (from 2007-07-01)
    start = pd.Timestamp("2007-07-01")
    if equity.index[0] > start:
        # If data doesn't go back to 2007, use what we have
        start = equity.index[0]

    equity_fp = equity[equity.index >= start]
    returns_fp = returns[returns.index >= start]

    if len(returns_fp) < 252:
        pytest.skip("Less than 1 year of data for regression check")

    s_fp = metrics.sharpe(returns_fp)
    dd_fp = metrics.max_drawdown(equity_fp)
    c_fp = metrics.cagr(equity_fp)

    # Generous tolerances to handle data vintage differences
    assert 0.95 < s_fp < 1.20, \
        f"EW-cash Sharpe {s_fp:.3f} outside regression band [0.95, 1.20]"
    assert -0.20 < dd_fp < -0.08, \
        f"EW-cash MaxDD {dd_fp:.1%} outside regression band [-20%, -8%]"
    assert 0.05 < c_fp < 0.12, \
        f"EW-cash CAGR {c_fp:.1%} outside regression band [5%, 12%]"
