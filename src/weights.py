"""Weighting functions.

Each function computes target weights for "in" assets (those that passed
the filter). Weights for "out" assets are zero. Weights do NOT need to
sum to 1.0 — the backtest engine handles renormalization based on
exit_mode.

Signature:
    def weight_name(prices: pd.DataFrame, in_mask: pd.DataFrame, **kwargs) -> pd.DataFrame

Adapted from trading/src/portfolio/weights.py, strategies.py, and
scripts/best_three_test.py.
"""

import numpy as np
import pandas as pd

from . import data as data_module


# ===================================================================
# PUBLIC API — these names are referenced in default.yaml
# ===================================================================

def equal(prices: pd.DataFrame, in_mask: pd.DataFrame,
          **kwargs) -> pd.DataFrame:
    """Equal weight among surviving (in_mask==True) assets.

    Each "in" asset gets 1/n_total (not 1/n_active). This means when
    assets are filtered out, total weight < 1.0 and the remainder is
    available for cash allocation by the backtest engine.

    Args:
        prices: Daily adjusted close prices, columns = tickers.
        in_mask: Boolean DataFrame from filter. True = asset is "in".

    Returns:
        Weight DataFrame. Each "in" asset gets 1/n_total. "Out" = 0.
    """
    n_total = len(prices.columns)
    target_weight = 1.0 / n_total
    weights = in_mask.astype(float) * target_weight
    return weights


def momentum(prices: pd.DataFrame, in_mask: pd.DataFrame,
             lookback_days: int = 252, skip_days: int = 21,
             split: list = None, **kwargs) -> pd.DataFrame:
    """Momentum-ranked weights among surviving assets.

    Ranks survivors by trailing 12-1 momentum (total return over
    lookback_days minus return over skip_days). Assigns weights
    from split list by rank (best to worst), then scales by
    n_survivors / n_total so that filtered-out assets leave weight
    available for cash allocation.

    Args:
        prices: Daily adjusted close prices.
        in_mask: Boolean DataFrame from filter.
        lookback_days: Total momentum lookback (default 252 = 12 months).
        skip_days: Recent period to skip (default 21 = 1 month).
        split: Weight allocation by rank, e.g. [0.70, 0.20, 0.10].
            Must have len(split) == number of tickers. If fewer survivors
            than split entries, top survivors get proportionally more.

    Returns:
        Weight DataFrame. Ranked by momentum, weights from split,
        scaled so total weight reflects fraction of universe surviving.
    """
    if split is None:
        split = [0.70, 0.20, 0.10]

    tickers = list(prices.columns)
    n_total = len(tickers)

    # Pre-compute rank weights for each possible number of survivors.
    # These sum to 1.0 among survivors, then we scale by n/n_total.
    rank_weights = {}
    for n in range(1, n_total + 1):
        if n <= len(split):
            # Fewer or equal survivors to split entries: take top n, renormalize
            rw = split[:n]
            total = sum(rw)
            rw = [w / total for w in rw] if total > 0 else [1.0 / n] * n
        else:
            # More survivors than split entries: pad extras with the smallest
            # split value, then renormalize. E.g. split=[0.70,0.20,0.10] with
            # 5 survivors → [0.70, 0.20, 0.10, 0.10, 0.10], normalized.
            rw = list(split) + [split[-1]] * (n - len(split))
            total = sum(rw)
            rw = [w / total for w in rw] if total > 0 else [1.0 / n] * n
        # Scale: if n < n_total, total weight < 1.0 → remainder = cash
        scale = n / n_total
        rank_weights[n] = [w * scale for w in rw]

    # Compute 12-1 momentum: return from t-lookback to t-skip
    # This is the standard academic momentum signal: total return over the
    # lookback period, excluding the most recent skip period.
    total_ret = prices / prices.shift(lookback_days) - 1
    recent_ret = prices / prices.shift(skip_days) - 1
    mom = (1 + total_ret) / (1 + recent_ret) - 1

    # Evaluate monthly (same as filter frequency)
    monthly_idx = data_module.month_end_dates(prices.index)
    eval_dates = sorted(set(monthly_idx) & set(prices.index))

    # Align in_mask to prices index
    invested = in_mask.reindex(prices.index).ffill().fillna(True)

    weights = pd.DataFrame(np.nan, index=prices.index, columns=tickers)

    # Only iterate monthly evaluation dates (~216 for 18yr), not every
    # trading day (~4500). Forward-fill weights to daily afterward.
    for dt in eval_dates:
        survivors = [t for t in tickers if invested.loc[dt, t]]
        n = len(survivors)
        if n == 0:
            weights.loc[dt] = 0.0
            continue

        scores = mom.loc[dt, survivors].dropna()
        if scores.empty:
            # No momentum data yet: fall back to equal weight (scaled)
            w = 1.0 / n_total
            for t in survivors:
                weights.loc[dt, t] = w
            for t in tickers:
                if t not in survivors:
                    weights.loc[dt, t] = 0.0
            continue

        ranked = scores.sort_values(ascending=False).index.tolist()
        rw = rank_weights[len(ranked)]
        for i, t in enumerate(ranked):
            weights.loc[dt, t] = rw[i]
        for t in tickers:
            if t not in ranked:
                weights.loc[dt, t] = 0.0

    # Forward-fill monthly evaluations to daily, fill leading NaNs with 0
    weights = weights.ffill().fillna(0.0)

    # Zero out weights for assets that are filtered out on any given day.
    # The filter mask changes daily but weights are only re-evaluated monthly,
    # so an asset could be "in" at evaluation but filtered out mid-month.
    weights = weights * invested.astype(float)

    return weights


def fixed(prices: pd.DataFrame, in_mask: pd.DataFrame,
          allocation: dict = None, **kwargs) -> pd.DataFrame:
    """Fixed weights from config, masked by filter.

    Args:
        prices: Daily adjusted close prices.
        in_mask: Boolean DataFrame from filter.
        allocation: Dict of {ticker: weight}, e.g. {"SPY": 0.70, "TLT": 0.20, "GLD": 0.10}.

    Returns:
        Weight DataFrame. Fixed target weights, zeroed for "out" assets.
    """
    if allocation is None:
        raise ValueError("fixed() requires 'allocation' dict in config")

    tickers = list(prices.columns)
    weights = pd.DataFrame(0.0, index=prices.index, columns=tickers)
    for t in tickers:
        if t in allocation:
            weights[t] = allocation[t] * in_mask[t].astype(float)

    return weights


def risk_parity(prices: pd.DataFrame, in_mask: pd.DataFrame,
                vol_lookback: int = 63, **kwargs) -> pd.DataFrame:
    """Inverse-volatility risk parity weights among surviving assets.

    Weights are scaled so that the total weight equals n_survivors / n_total.
    When assets are filtered out, the remainder is available for cash.

    Args:
        prices: Daily adjusted close prices.
        in_mask: Boolean DataFrame from filter.
        vol_lookback: Rolling window for volatility estimation.

    Returns:
        Weight DataFrame. Inverse-vol weighted, zeroed for "out" assets.
        Total weight < 1.0 when assets are filtered out.
    """
    n_total = len(prices.columns)
    returns = prices.pct_change()
    vol = returns.rolling(window=vol_lookback, min_periods=21).std() * np.sqrt(252)

    # Mask out "out" assets before computing inverse-vol
    masked_inv_vol = (1.0 / vol) * in_mask.astype(float)
    masked_inv_vol = masked_inv_vol.replace([np.inf, -np.inf], np.nan)

    # Normalize among survivors to sum to 1.0, then scale by n_surv/n_total
    row_sums = masked_inv_vol.sum(axis=1).replace(0, np.nan)
    rp_weights = masked_inv_vol.div(row_sums, axis=0).fillna(0.0)

    # Scale: total weight reflects fraction of universe surviving
    n_active = in_mask.astype(float).sum(axis=1)
    scale = n_active / n_total
    rp_weights = rp_weights.mul(scale, axis=0)

    return rp_weights
