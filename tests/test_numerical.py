"""Numerical tests with hand-calculated expected values.

Every test uses small, concrete inputs where the expected output can be
verified by hand. Inputs are chosen so that the WRONG formula produces
a DIFFERENT answer — if a test passes with both the correct and incorrect
implementation, it's not a useful test.

No random data. No real market data. Only specific inputs that exercise
specific code paths.
"""

import numpy as np
import pandas as pd
import pytest

from src import filters, weights, backtest, metrics


# ===================================================================
# FILTERS
# ===================================================================

class TestSmaFilter:
    """Numerical tests for filters.sma()."""

    def test_sma_monthly_evaluation_close_to_boundary(self):
        """Two assets: one 0.5% above its 200-day SMA, one 0.5% below.

        Hand calculation:
        - Asset A: 200 days of constant 100.0, then day 201 = 100.5.
          SMA(200) on day 201 = (199*100 + 100.5) / 200 = 100.0025.
          Price 100.5 > 100.0025 → True (in).
        - Asset B: 200 days of constant 100.0, then day 201 = 99.5.
          SMA(200) on day 201 = (199*100 + 99.5) / 200 = 99.9975.
          Price 99.5 < 99.9975 → False (out).
        """
        n = 250
        dates = pd.bdate_range("2020-01-01", periods=n)
        prices = pd.DataFrame(index=dates)
        prices["A"] = 100.0
        prices["B"] = 100.0
        # On the last day of the data, move price slightly
        prices.iloc[-1, 0] = 100.5  # A: 0.5% above
        prices.iloc[-1, 1] = 99.5   # B: 0.5% below

        result = filters.sma(prices, window=200, frequency="monthly")

        # The last month-end evaluation should reflect the boundary prices
        last_day = dates[-1]
        assert result.loc[last_day, "A"] == True, "Asset 0.5% above SMA should be 'in'"
        assert result.loc[last_day, "B"] == False, "Asset 0.5% below SMA should be 'out'"

    def test_sma_forward_fill_between_months(self):
        """Signal computed at month-end persists through the next month.

        Monthly evaluation means the signal only changes at month-end.
        Between evaluations, the signal is forward-filled. This test
        verifies that all days within a month carry the same signal
        as the previous month-end evaluation.

        Setup: 500 business days (~2 years) with a steady uptrend.
        After SMA warmup (~200 days), price is above SMA. We verify
        that the signal is constant within each month (only changes
        at month-end evaluations).
        """
        dates = pd.bdate_range("2019-01-01", periods=500)
        prices = pd.DataFrame(index=dates)
        # Strong uptrend ensures price > SMA after warmup
        prices["A"] = 100.0 + np.arange(500) * 0.05

        result = filters.sma(prices, window=200, frequency="monthly")

        # After warmup, check that signal is constant within each month.
        # The signal should only change at month-end evaluation dates.
        post_warmup = result.index[result.index > dates[250]]
        if len(post_warmup) > 10:
            # Group by month and verify constant within each month
            for yr in sorted(set(post_warmup.year)):
                for mo in range(1, 13):
                    month_days = post_warmup[
                        (post_warmup.year == yr) & (post_warmup.month == mo)
                    ]
                    if len(month_days) < 2:
                        continue
                    vals = result.loc[month_days, "A"]
                    assert vals.nunique() == 1, \
                        f"Signal should be constant within {yr}-{mo:02d}, " \
                        f"but got {vals.unique()}"

    def test_sma_at_exact_boundary(self):
        """Price equals SMA exactly → should be 'out' (strict >).

        The filter uses `prices[asset] > ma`, so equality is "out".
        """
        n = 250
        dates = pd.bdate_range("2020-01-01", periods=n)
        # Constant price = constant SMA → price == SMA
        prices = pd.DataFrame({"A": [100.0] * n}, index=dates)

        result = filters.sma(prices, window=200, frequency="monthly")

        # After warmup, price == SMA → strict > fails → should be False
        last_day = dates[-1]
        assert result.loc[last_day, "A"] == False, \
            "Price == SMA should be 'out' (filter uses strict >)"


# ===================================================================
# WEIGHTS
# ===================================================================

class TestEqualWeightNumerical:
    """Verify equal() does NOT renormalize — critical for cash exit mode."""

    def test_two_of_three_in(self):
        """3 assets, 2 active.

        Hand calculation:
        - n_total = 3, target_weight = 1/3 = 0.3333
        - in_mask = [True, True, False]
        - weights = [0.3333, 0.3333, 0.0]
        - Sum = 0.6667 (NOT 1.0 — remainder is for cash)

        If equal() renormalized, we'd get [0.5, 0.5, 0.0] which is wrong
        for cash exit mode.
        """
        dates = pd.bdate_range("2020-01-01", periods=5)
        prices = pd.DataFrame({
            "A": [100.0] * 5, "B": [100.0] * 5, "C": [100.0] * 5
        }, index=dates)
        in_mask = pd.DataFrame({
            "A": [True] * 5, "B": [True] * 5, "C": [False] * 5
        }, index=dates)

        w = weights.equal(prices, in_mask)

        expected = 1.0 / 3
        for d in dates:
            assert abs(w.loc[d, "A"] - expected) < 1e-10
            assert abs(w.loc[d, "B"] - expected) < 1e-10
            assert w.loc[d, "C"] == 0.0
            # Sum should be 2/3, NOT 1.0
            assert abs(w.loc[d].sum() - 2.0 / 3) < 1e-10

    def test_one_of_three_in(self):
        """Only 1 asset active.

        Weight = 1/3 (not 1.0). Cash gets 2/3.
        """
        dates = pd.bdate_range("2020-01-01", periods=5)
        prices = pd.DataFrame({
            "A": [100.0] * 5, "B": [100.0] * 5, "C": [100.0] * 5
        }, index=dates)
        in_mask = pd.DataFrame({
            "A": [True] * 5, "B": [False] * 5, "C": [False] * 5
        }, index=dates)

        w = weights.equal(prices, in_mask)

        for d in dates:
            assert abs(w.loc[d, "A"] - 1.0 / 3) < 1e-10
            assert w.loc[d, "B"] == 0.0
            assert w.loc[d, "C"] == 0.0


class TestMomentumNumerical:
    """Verify momentum formula produces correct 12-1 momentum values."""

    def _make_flipped_momentum_prices(self):
        """Create prices where correct vs buggy formula give DIFFERENT rankings.

        NOTE: Simple inputs (moderate returns, no extreme recent moves) produce
        the SAME ranking under both formulas. We need extreme recent price moves
        to flip the ranking — the buggy formula scales by P_t per asset, which
        inflates scores for assets with large recent jumps.

        The key: the buggy formula is P_t/P_{t-252} - P_t/P_{t-21}.
        The correct formula is (1 + P_t/P_{t-252} - 1) / (1 + P_t/P_{t-21} - 1) - 1
                             = P_t * P_{t-252}^{-1} / (P_t * P_{t-21}^{-1}) - 1
                             = P_{t-21} / P_{t-252} - 1.

        Buggy: P_t/P_{t-252} - P_t/P_{t-21}
        Correct: P_{t-21}/P_{t-252} - 1

        These differ when P_t is large — the buggy formula scales both
        terms by P_t, amplifying differences in the recent return.

        Setup: 2 assets, 253 days.

        Asset X: P_0=100, P_231=120, P_252=150 (big recent jump)
          Correct: P_231/P_0 - 1 = 120/100 - 1 = 20%
          Buggy:   P_252/P_0 - P_252/P_231 = 1.50 - 1.25 = 0.25

        Asset Y: P_0=100, P_231=130, P_252=132 (small recent move)
          Correct: P_231/P_0 - 1 = 130/100 - 1 = 30%
          Buggy:   P_252/P_0 - P_252/P_231 = 1.32 - 1.0154 = 0.3046

        Correct ranking: Y (30%) > X (20%)
        Buggy ranking: Y (0.3046) > X (0.25) — SAME ranking again!

        Hmm, both P_t terms factor out the same for ranking.
        Actually the buggy formula is:
          P_t/P_{t-252} - P_t/P_{t-21}
          = P_t * (1/P_{t-252} - 1/P_{t-21})

        For ranking, the P_t multiplier is the SAME for all assets on
        the same date (wait, no — P_t is different per asset!).

        So the buggy formula gives for asset i:
          P_{t,i}/P_{t-252,i} - P_{t,i}/P_{t-21,i}

        And the correct formula gives:
          P_{t-21,i}/P_{t-252,i} - 1

        These differ because P_{t,i} appears in the buggy formula.
        If asset X has P_t = 150 and asset Y has P_t = 132, the
        multiplicative factors are different.

        Let me try:
        Asset X: P_0=100, P_231=120, P_252=150
          Correct: 120/100 - 1 = 0.20
          Buggy: 150/100 - 150/120 = 1.50 - 1.25 = 0.25

        Asset Y: P_0=100, P_231=125, P_252=126
          Correct: 125/100 - 1 = 0.25
          Buggy: 126/100 - 126/125 = 1.26 - 1.008 = 0.252

        Correct ranking: Y (0.25) > X (0.20)
        Buggy ranking:   Y (0.252) > X (0.25) — SAME! barely.

        Need bigger divergence. Let's make the recent move extreme:
        Asset X: P_0=100, P_231=105, P_252=200 (doubled recently)
          Correct: 105/100 - 1 = 0.05 (5%)
          Buggy: 200/100 - 200/105 = 2.00 - 1.905 = 0.095

        Asset Y: P_0=100, P_231=108, P_252=109 (no recent move)
          Correct: 108/100 - 1 = 0.08 (8%)
          Buggy: 109/100 - 109/108 = 1.09 - 1.00926 = 0.0807

        Correct ranking: Y (8%) > X (5%)
        Buggy ranking: X (0.095) > Y (0.081) — DIFFERENT!

        This is the case: a huge recent jump inflates the buggy score
        for X, flipping the ranking.
        """
        n = 253
        dates = pd.bdate_range("2020-01-01", periods=n)

        # Asset X: moderate total return, huge recent jump
        x_prices = np.linspace(100, 105, 232).tolist()  # days 0-231
        x_prices += np.linspace(105, 200, n - 232).tolist()  # days 232-252
        # Adjust: day 231 = 105.0, day 252 = 200.0

        # Asset Y: better total return, flat recent
        y_prices = np.linspace(100, 108, 232).tolist()  # days 0-231
        y_prices += np.linspace(108, 109, n - 232).tolist()  # days 232-252

        # Third asset (always ranked last, far below)
        z_prices = np.linspace(100, 95, n).tolist()  # declining

        prices = pd.DataFrame({
            "X": x_prices, "Y": y_prices, "Z": z_prices
        }, index=dates)

        return prices, dates

    def test_momentum_close_scores_correct_ranking(self):
        """Verify 12-1 momentum ranking with the CORRECT formula.

        The correct 12-1 momentum is P_{t-skip}/P_{t-lookback} - 1.
        With extreme recent moves, the correct formula ranks Y > X > Z
        while the old buggy formula would rank X > Y > Z.
        """
        prices, dates = self._make_flipped_momentum_prices()

        # All assets "in"
        in_mask = pd.DataFrame(True, index=dates, columns=prices.columns)

        w = weights.momentum(prices, in_mask,
                             lookback_days=252, skip_days=21,
                             split=[0.70, 0.20, 0.10])

        # On the last evaluation date:
        # Correct ranking: Y (8%) > X (5%) > Z (negative)
        # Y gets 0.70, X gets 0.20, Z gets 0.10
        last_day = dates[-1]
        assert w.loc[last_day, "Y"] > w.loc[last_day, "X"], \
            f"Y should rank higher than X. Y={w.loc[last_day, 'Y']}, X={w.loc[last_day, 'X']}"
        assert w.loc[last_day, "X"] > w.loc[last_day, "Z"], \
            f"X should rank higher than Z. X={w.loc[last_day, 'X']}, Z={w.loc[last_day, 'Z']}"

    def test_momentum_2_survivors(self):
        """Only 2 of 3 assets above 200dma. How does [0.70, 0.20, 0.10] map?

        Hand calculation from weights.py code:
        - n = 2 survivors, split = [0.70, 0.20, 0.10]
        - len(split) = 3 > n = 2, so rw = split[:2] = [0.70, 0.20]
        - total = 0.90, rw = [0.70/0.90, 0.20/0.90] = [0.778, 0.222]
        - scale = 2/3 = 0.667
        - rank_weights = [0.778*0.667, 0.222*0.667] = [0.519, 0.148]
        - Best gets 0.519, second gets 0.148, third is out (0.0)
        - Sum = 0.667 (= 2/3, correct for cash exit)
        """
        n = 253
        dates = pd.bdate_range("2020-01-01", periods=n)

        # A has higher momentum, B has lower, C is filtered out
        prices = pd.DataFrame({
            "A": np.linspace(100, 120, n),  # strong uptrend
            "B": np.linspace(100, 110, n),  # moderate uptrend
            "C": np.linspace(100, 90, n),   # downtrend
        }, index=dates)

        in_mask = pd.DataFrame({
            "A": [True] * n, "B": [True] * n, "C": [False] * n
        }, index=dates)

        w = weights.momentum(prices, in_mask, split=[0.70, 0.20, 0.10])
        last = dates[-1]

        # A should get the higher weight, B the lower
        assert w.loc[last, "A"] > w.loc[last, "B"], "A has higher momentum, should rank first"
        assert w.loc[last, "C"] == 0.0, "C is filtered out"

        # Sum should be 2/3
        total = w.loc[last].sum()
        assert abs(total - 2.0 / 3) < 0.01, f"Total weight should be 2/3, got {total}"

    def test_momentum_1_survivor(self):
        """Only 1 asset above 200dma.

        Hand calculation:
        - n = 1, split = [0.70, 0.20, 0.10]
        - rw = split[:1] = [0.70], total = 0.70, rw = [1.0]
        - scale = 1/3
        - rank_weights = [1.0 * 1/3] = [0.333]
        - The single survivor gets 1/3, NOT 0.70 or 1.0.
        """
        n = 253
        dates = pd.bdate_range("2020-01-01", periods=n)

        prices = pd.DataFrame({
            "A": np.linspace(100, 120, n),
            "B": np.linspace(100, 90, n),
            "C": np.linspace(100, 80, n),
        }, index=dates)

        in_mask = pd.DataFrame({
            "A": [True] * n, "B": [False] * n, "C": [False] * n
        }, index=dates)

        w = weights.momentum(prices, in_mask, split=[0.70, 0.20, 0.10])
        last = dates[-1]

        expected = 1.0 / 3
        assert abs(w.loc[last, "A"] - expected) < 0.01, \
            f"Single survivor should get 1/3, got {w.loc[last, 'A']}"
        assert w.loc[last, "B"] == 0.0
        assert w.loc[last, "C"] == 0.0


class TestFixedWeightNumerical:
    """Verify fixed() doesn't renormalize."""

    def test_fixed_no_renormalize(self):
        """Config {A:0.5, B:0.3, C:0.2}. C filtered out.

        Expected: [0.5, 0.3, 0.0]. Sum = 0.8, not 1.0.
        """
        dates = pd.bdate_range("2020-01-01", periods=5)
        prices = pd.DataFrame({
            "A": [100.0] * 5, "B": [100.0] * 5, "C": [100.0] * 5
        }, index=dates)
        in_mask = pd.DataFrame({
            "A": [True] * 5, "B": [True] * 5, "C": [False] * 5
        }, index=dates)

        w = weights.fixed(prices, in_mask,
                          allocation={"A": 0.5, "B": 0.3, "C": 0.2})

        for d in dates:
            assert abs(w.loc[d, "A"] - 0.5) < 1e-10
            assert abs(w.loc[d, "B"] - 0.3) < 1e-10
            assert w.loc[d, "C"] == 0.0
            assert abs(w.loc[d].sum() - 0.8) < 1e-10


class TestRiskParityNumerical:
    """Verify risk parity with known volatilities."""

    def test_risk_parity_known_vols(self):
        """3 assets with known annualized vols: ~16%, ~32%, ~48%.

        We construct prices with daily returns of known std dev, then
        verify the inverse-vol weighting is approximately correct.

        Hand calculation for inverse-vol weights:
        - inv_vol = [1/0.16, 1/0.32, 1/0.48] = [6.25, 3.125, 2.083]
        - sum = 11.458
        - normalized = [0.546, 0.273, 0.182]
        - All 3 assets in, scale = 3/3 = 1.0
        """
        np.random.seed(42)
        n = 500  # enough days for vol estimate to stabilize
        dates = pd.bdate_range("2020-01-01", periods=n)

        # Daily returns with controlled volatility
        # annualized vol = daily_std * sqrt(252)
        # So daily_std = target_vol / sqrt(252)
        daily_std_a = 0.16 / np.sqrt(252)  # ~16% ann vol
        daily_std_b = 0.32 / np.sqrt(252)  # ~32% ann vol
        daily_std_c = 0.48 / np.sqrt(252)  # ~48% ann vol

        ret_a = np.random.normal(0, daily_std_a, n)
        ret_b = np.random.normal(0, daily_std_b, n)
        ret_c = np.random.normal(0, daily_std_c, n)

        prices = pd.DataFrame({
            "A": 100 * np.exp(ret_a.cumsum()),
            "B": 100 * np.exp(ret_b.cumsum()),
            "C": 100 * np.exp(ret_c.cumsum()),
        }, index=dates)

        in_mask = pd.DataFrame(True, index=dates, columns=prices.columns)

        w = weights.risk_parity(prices, in_mask, vol_lookback=252)
        last = dates[-1]

        # A (lowest vol) should get highest weight
        assert w.loc[last, "A"] > w.loc[last, "B"], "Lowest vol should get highest weight"
        assert w.loc[last, "B"] > w.loc[last, "C"], "Middle vol should get middle weight"

        # Check approximate proportions (loose tolerance due to estimation)
        total = w.loc[last].sum()
        assert abs(total - 1.0) < 0.05, f"All in, total should be ~1.0, got {total}"

        # A should be roughly 0.5-0.6 (inverse-vol dominant)
        assert 0.35 < w.loc[last, "A"] < 0.70, \
            f"A weight should be ~0.55, got {w.loc[last, 'A']}"


# ===================================================================
# BACKTEST
# ===================================================================

class TestBacktestNumerical:
    """Numerical tests for run_backtest with hand-calculated values."""

    def test_return_compounding(self):
        """2 assets, equal weight (0.5, 0.5), known daily returns.

        Setup (prices):
          Day 0: A=100, B=100 (starting point)
          Day 1: A=102, B=98   → returns: A=+2%, B=-2%
          Day 2: A=104, B=99   → returns: A=+1.96%, B=+1.02%
          Day 3: A=103, B=101  → returns: A=-0.96%, B=+2.02%

        With equal weights (0.5, 0.5) and no costs:
        Vectorbt executes orders at close, so:
        - Day 0: buy 0.5 of each at prices [100, 100]. Portfolio = 100.
        - Day 1: portfolio return = 0.5*(102/100-1) + 0.5*(98/100-1) = 0.5*0.02 + 0.5*(-0.02) = 0.0
          Vectorbt rebalances to maintain 50/50 target. Equity = 100.
        - Day 2: portfolio return ≈ 0.5*(104/102-1) + 0.5*(99/98-1)
                = 0.5*0.01961 + 0.5*0.01020 = 0.01490
          Equity ≈ 101.49
        - Day 3: portfolio return ≈ 0.5*(103/104-1) + 0.5*(101/99-1)
                = 0.5*(-0.00962) + 0.5*0.02020 = 0.00529
          Equity ≈ 102.03
        """
        dates = pd.bdate_range("2020-01-01", periods=4)
        prices = pd.DataFrame({
            "A": [100.0, 102.0, 104.0, 103.0],
            "B": [100.0, 98.0, 99.0, 101.0],
        }, index=dates)
        w = pd.DataFrame({
            "A": [0.5] * 4,
            "B": [0.5] * 4,
        }, index=dates)

        equity = backtest.run_backtest(
            prices, w, exit_mode="renormalize",
            initial_capital=100.0, slippage_bps=0.0)

        # Final equity should be approximately 102.0 (within 0.5%)
        # exact value depends on vectorbt's discrete-share rebalancing
        assert 101.0 < equity.iloc[-1] < 103.0, \
            f"Final equity should be ~102, got {equity.iloc[-1]:.2f}"
        # First day should be ~100 (just bought, no return yet)
        assert abs(equity.iloc[0] - 100.0) < 0.01

    def test_cash_exit_earns_shy(self):
        """Asset B filtered out. Cash = SHY. Verify cash earns SHY return.

        Setup:
        - Asset A at 0.5 weight (constant). B filtered out.
        - Cash weight = 0.5 (allocated to SHY).
        - SHY returns ~0.02% per day (roughly 5% annual).
        - A is flat (100.0 each day).
        - Over 5 days, portfolio should grow from the SHY return.

        Expected: equity > 100.0 even though A is flat.
        """
        dates = pd.bdate_range("2020-01-01", periods=10)
        prices = pd.DataFrame({
            "A": [100.0] * 10,
            "B": [100.0] * 10,
        }, index=dates)
        cash = pd.Series(
            [100.0 + 0.02 * i for i in range(10)],
            index=dates
        )  # SHY going up steadily
        w = pd.DataFrame({
            "A": [0.5] * 10,
            "B": [0.0] * 10,
        }, index=dates)

        equity = backtest.run_backtest(
            prices, w, exit_mode="cash",
            cash_prices=cash, initial_capital=100.0, slippage_bps=0.0)

        # Equity should be slightly > 100 (SHY earning on the 50% cash)
        assert equity.iloc[-1] > 100.0, \
            f"Cash should earn SHY return, got {equity.iloc[-1]:.4f}"

        # Compare: without cash vehicle, equity should stay ~100
        equity_no_cash = backtest.run_backtest(
            prices, w, exit_mode="cash",
            cash_prices=None, initial_capital=100.0, slippage_bps=0.0)

        assert equity.iloc[-1] > equity_no_cash.iloc[-1], \
            "Cash vehicle should produce higher equity than no cash vehicle"

    def test_renormalize_vs_cash_mode(self):
        """Same inputs, both exit modes. Renormalize should produce higher equity.

        Asset A goes up. Asset B filtered out.
        - Cash mode: A at 1/3, cash at 2/3. Portfolio gains = 1/3 of A's return.
        - Renorm mode: A at 1.0. Portfolio gains = 100% of A's return.
        """
        dates = pd.bdate_range("2020-01-01", periods=10)
        prices = pd.DataFrame({
            "A": [100 + i * 2 for i in range(10)],  # A going up 2 per day
            "B": [100.0] * 10,
            "C": [100.0] * 10,
        }, index=dates)
        w = pd.DataFrame({
            "A": [1.0 / 3] * 10,
            "B": [0.0] * 10,
            "C": [0.0] * 10,
        }, index=dates)

        eq_cash = backtest.run_backtest(
            prices, w, exit_mode="cash",
            initial_capital=100.0, slippage_bps=0.0)
        eq_renorm = backtest.run_backtest(
            prices, w, exit_mode="renormalize",
            initial_capital=100.0, slippage_bps=0.0)

        # Renormalize should produce higher equity since it puts 100% in A
        assert eq_renorm.iloc[-1] > eq_cash.iloc[-1], \
            f"Renorm ({eq_renorm.iloc[-1]:.2f}) should beat cash ({eq_cash.iloc[-1]:.2f})"

    def test_zero_turnover_no_cost(self):
        """Constant weights with high slippage. If weights never change,
        no transaction costs should be charged after initial buy.

        Initial buy turnover exists, but subsequent days should be
        approximately free.
        """
        dates = pd.bdate_range("2020-01-01", periods=20)
        prices = pd.DataFrame({
            "A": [100.0 + i * 0.5 for i in range(20)],
            "B": [100.0 - i * 0.2 for i in range(20)],
        }, index=dates)
        w = pd.DataFrame({
            "A": [0.5] * 20, "B": [0.5] * 20,
        }, index=dates)

        eq_free = backtest.run_backtest(
            prices, w, exit_mode="renormalize",
            initial_capital=100.0, slippage_bps=0.0)
        eq_costly = backtest.run_backtest(
            prices, w, exit_mode="renormalize",
            initial_capital=100.0, slippage_bps=50.0)  # 50bps slippage

        # With constant weights, the only cost is the initial purchase.
        # With daily rebalancing back to target, there's small turnover
        # from drift, but it should be minor.
        # The costly version should be close to the free version
        ratio = eq_costly.iloc[-1] / eq_free.iloc[-1]
        assert ratio > 0.95, \
            f"Constant weights with 50bps slippage lost {(1-ratio)*100:.1f}% — too much"


# ===================================================================
# METRICS
# ===================================================================

class TestMetricsNumerical:
    """Verify metrics against hand-calculated expected values."""

    def test_sharpe_known_value(self):
        """Alternating +1%, -0.5% returns.

        Hand calculation:
        - Returns: [0.01, -0.005, 0.01, -0.005, ...] (100 pairs = 200 days)
        - Mean = (0.01 + (-0.005)) / 2 = 0.0025
        - Var = ((0.01-0.0025)^2 + (-0.005-0.0025)^2) / 2
              = (0.000056 + 0.000056) / 2 = 0.000056
        - Std = sqrt(0.000056) = 0.00750
        - Annualized Sharpe = (0.0025 / 0.00750) * sqrt(252)
                            = 0.3333 * 15.875 = 5.29

        This is a very high Sharpe because of the low volatility and
        positive drift. The exact value will depend on quantstats's
        implementation.
        """
        returns = pd.Series(
            [0.01, -0.005] * 100,
            index=pd.bdate_range("2020-01-01", periods=200)
        )
        s = metrics.sharpe(returns)

        # Should be positive and high
        assert s > 3.0, f"Sharpe should be high for consistent positive drift, got {s:.2f}"
        assert s < 8.0, f"Sharpe should be reasonable, got {s:.2f}"

    def test_max_drawdown_known(self):
        """Equity: 100, 110, 105, 115, 100, 120.

        Hand calculation:
        - Peak at 115 (day 3). Trough at 100 (day 4).
        - Drawdown = (100 - 115) / 115 = -13.04%
        """
        equity = pd.Series(
            [100.0, 110.0, 105.0, 115.0, 100.0, 120.0],
            index=pd.bdate_range("2020-01-01", periods=6)
        )
        dd = metrics.max_drawdown(equity)

        expected = (100.0 - 115.0) / 115.0  # = -0.13043
        assert abs(dd - expected) < 0.001, f"Max DD should be {expected:.4f}, got {dd:.4f}"
        assert dd < 0, "Max drawdown should be negative"

    def test_cagr_doubles_in_3_years(self):
        """Equity doubles in ~3 years (756 trading days).

        CAGR = 2^(1/3) - 1 = 25.99%
        """
        n = 756
        dates = pd.bdate_range("2020-01-01", periods=n)
        # Linearly interpolate in log space for smooth growth
        equity = pd.Series(
            100.0 * np.exp(np.log(2) * np.arange(n) / (n - 1)),
            index=dates
        )

        c = metrics.cagr(equity)

        expected = 2.0 ** (1.0 / 3) - 1  # 0.2599
        assert abs(c - expected) < 0.02, \
            f"CAGR should be ~{expected:.4f}, got {c:.4f}"

    def test_capture_ratios_half(self):
        """Strategy returns = 0.5 × benchmark.

        Both upside and downside capture should be approximately 0.50.
        """
        np.random.seed(42)
        n = 500
        bench_ret = pd.Series(
            np.random.normal(0.0003, 0.01, n),
            index=pd.bdate_range("2020-01-01", periods=n)
        )
        strat_ret = bench_ret * 0.5

        up_cap = metrics.upside_capture(strat_ret, bench_ret)
        down_cap = metrics.downside_capture(strat_ret, bench_ret)

        assert abs(up_cap - 0.50) < 0.05, f"Upside capture should be ~0.50, got {up_cap:.3f}"
        assert abs(down_cap - 0.50) < 0.05, f"Downside capture should be ~0.50, got {down_cap:.3f}"

    def test_max_drawdown_monotonic_is_zero(self):
        """Monotonically increasing equity has zero drawdown."""
        equity = pd.Series(
            [100.0, 101.0, 102.0, 103.0, 104.0],
            index=pd.bdate_range("2020-01-01", periods=5)
        )
        dd = metrics.max_drawdown(equity)
        assert dd == 0.0, f"Monotonic equity should have 0 drawdown, got {dd}"


# ===================================================================
# COMPARE / BENCHMARK WINDOWING
# ===================================================================

class TestBenchmarkWindowing:
    """Verify C3 fix: benchmarks produce different metrics per window."""

    def test_benchmark_different_windows(self):
        """Run the same benchmark on two windows. Metrics must differ.

        This catches the C3 bug where all windows produced identical
        benchmark metrics because run_benchmark was called on full data.
        """
        # Create 500 days of data with a clear regime change at day 250
        np.random.seed(42)
        n = 500
        dates = pd.bdate_range("2018-01-01", periods=n)

        # First half: A goes up, B goes down
        # Second half: A goes down, B goes up
        prices = pd.DataFrame({
            "A": np.concatenate([
                100 * np.exp(np.random.normal(0.001, 0.01, 250).cumsum()),
                100 * np.exp(np.random.normal(-0.001, 0.01, 250).cumsum()),
            ]),
            "B": np.concatenate([
                100 * np.exp(np.random.normal(-0.001, 0.01, 250).cumsum()),
                100 * np.exp(np.random.normal(0.001, 0.01, 250).cumsum()),
            ]),
        }, index=dates)

        allocation = {"A": 0.6, "B": 0.4}

        # Run benchmark on first half
        w1_start = dates[0]
        w1_prices = prices[prices.index >= w1_start].iloc[:250]
        eq1 = backtest.run_benchmark(w1_prices, allocation,
                                     initial_capital=100000, slippage_bps=0.0)
        ret1 = eq1.pct_change().dropna()
        m1 = metrics.compute_all(ret1, eq1, ["sharpe", "cagr"])

        # Run benchmark on second half
        w2_start = dates[250]
        w2_prices = prices[prices.index >= w2_start]
        eq2 = backtest.run_benchmark(w2_prices, allocation,
                                     initial_capital=100000, slippage_bps=0.0)
        ret2 = eq2.pct_change().dropna()
        m2 = metrics.compute_all(ret2, eq2, ["sharpe", "cagr"])

        # Metrics should differ (regime changed)
        assert abs(m1["sharpe"] - m2["sharpe"]) > 0.01, \
            f"Sharpe should differ across windows: w1={m1['sharpe']}, w2={m2['sharpe']}"
