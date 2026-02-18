# The 200dma Strategy: A Three-Fund Approach to Not Losing Money

## The Punchline

After testing inverse-volatility weighting, momentum rotation, dual momentum, hybrid filters, fixed tilts, and a dozen other variations across 18 years of data, the entire design space collapsed into a 2x2 matrix:

|  | Cash Exit | Renormalize |
|---|---|---|
| **Equal Weight** | 1.04 Sharpe, 7.9% CAGR, -14% MaxDD | 0.98 Sharpe, 11.3% CAGR, -22% MaxDD |
| **Momentum Tilt** | 0.98 Sharpe, 9.4% CAGR, -17% MaxDD | 0.98 Sharpe, 13.1% CAGR, -21% MaxDD |

---

## What Is This?

Three funds: SPY (US stocks), TLT (long-term bonds), GLD (gold). One signal: is each fund above or below its 200-day moving average? Check once a month, on the last trading day. That's the whole strategy.

The equal-weight cash-exit variant (EW-cash) is essentially Meb Faber's timing model applied to a three-fund portfolio. Faber first published the idea in 2006 as "A Simple Approach to Market Timing," then formalized it in 2007 as ["A Quantitative Approach to Tactical Asset Allocation"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=962461) in *The Journal of Wealth Management*. His insight: a 10-month simple moving average (roughly equivalent to a 200-day SMA) applied to asset classes reduces drawdowns without sacrificing much long-term return. Our EW-cash strategy is a direct implementation of this idea — equal weight across three diversified asset classes, exit to cash when below trend, rebalance monthly. The other three cells in the 2x2 matrix (renormalize, momentum tilt) are extensions that explore what happens when you stay fully invested or add momentum ranking on top of Faber's base filter.

## The Two Decisions

### Decision 1: What happens when a fund drops below its 200dma?

**Cash exit**: That fund's share goes to cash. If GLD drops below trend, you go from 33/33/33 to 33/33/0 with 33% in cash. Total portfolio exposure drops. You give up returns but dramatically limit drawdowns.

**Renormalize**: Redistribute the dead fund's weight to the survivors. If GLD drops, you go from 33/33/33 to 50/50/0. You stay ~100% invested at all times. More returns, deeper drawdowns. The risk: when two funds are below trend, renormalization puts you 100% in a single asset. In 2022, this briefly meant 100% in whichever fund was falling the slowest.

The numbers:

| | Cash Exit | Renormalize | Delta |
|---|---|---|---|
| CAGR | 7.9% | 11.2% | +3.3% |
| Max Drawdown | -13.8% | -22.6% | -8.8% |
| Sharpe | 1.038 | 0.962 | -0.076 |
| $200K becomes | $918K | $1.61M | +$690K |

Renormalize nearly doubles your terminal wealth. But it comes with drawdowns that are 64% deeper. Sharpe drops by ~0.08 — the extra return comes with proportionally more risk. You're dialing the risk/return knob, not getting a free lunch.

### The redistribution dial

Faber's paper treats exit-to-cash as a given. But what if you don't exit entirely to cash? What if you redistribute some of the freed capital to the surviving assets?

It turns out this is a continuous parameter — `redistribution_pct` — that goes from 0% (pure cash exit, Faber's version) to 100% (full renormalize). At 50%, half of the freed weight goes back to survivors and half stays in cash. We swept this parameter in 5% increments across the full 2007–2026 period:

| Redistribution | Sharpe | CAGR | Max Drawdown |
|---|---|---|---|
| 0% (cash) | 1.038 | 7.9% | -13.8% |
| 25% | 1.030 | 8.7% | -15.2% |
| 50% | 1.010 | 9.6% | -17.4% |
| 75% | 0.987 | 10.4% | -19.6% |
| 100% (renorm) | 0.962 | 11.2% | -22.6% |

Sharpe stays within a ~0.08 band across the entire range. CAGR climbs monotonically from 7.9% to 11.2%. Max drawdown worsens monotonically from -13.8% to -22.6%. For equal weight, this monotonic pattern holds across every time window we tested. Momentum tells a different story.

This means the cash-vs-renormalize choice isn't an optimization — it's a risk appetite dial. You're choosing how much drawdown you can stomach in exchange for higher compounding. There's no free lunch hidden at any point along the curve. Pick the drawdown you can sleep through, set the parameter, and move on.

The full sweep data (21 data points × 4 configurations) is in `results/redistribution_sweep*.csv`.

### The momentum exception

Run the same sweep with momentum weights instead of equal weight:

| Redistribution | Sharpe | CAGR | Max Drawdown |
|---|---|---|---|
| 0% (cash) | 0.983 | 9.4% | -16.9% |
| 25% | **0.993** | 10.3% | **-15.4%** |
| 50% | 0.991 | 11.2% | -16.8% |
| 75% | 0.982 | 12.1% | -18.8% |
| 100% (renorm) | 0.966 | 12.9% | -21.7% |

At 25% redistribution, max drawdown *improves* from -16.9% to -15.4% — while CAGR rises from 9.4% to 10.3%. Sharpe peaks at 0.994 around 25–30%. This is not a monotonic dial. There's a genuine sweet spot.

The mechanism: momentum weights (70/20/10) concentrate freed capital in the top-momentum survivor. During the GFC, when SPY collapsed below its 200dma, the highest-momentum survivor was TLT — the flight-to-safety winner. At 25% redistribution, a quarter of SPY's freed weight flows to survivors, heavily concentrated in TLT. The other 75% stays in cash. You get enough TLT exposure to capture its crisis rally while keeping most of the portfolio protected. Equal weight can't do this — it redistributes 50/50 to survivors, with no concentration in the winner.

Why doesn't this show up post-COVID? Because there's no sustained crisis where one asset is clearly the safe haven. Rising rates made TLT the worst performer in 2022, not the best. Momentum still concentrates in the top survivor, but that survivor no longer correlates with drawdown protection. Post-COVID, max drawdown worsens monotonically at every redistribution level — same as equal weight. Sharpe still peaks at ~25% (returns rise faster than volatility in that range), but the drawdown sweet spot is gone.

The takeaway: the ~25% sweet spot is regime-dependent. It works when momentum correctly identifies the safe-haven asset during a crisis — which it did during the GFC but not during the 2022 rate-hiking drawdown. Don't optimize to it.

### Decision 2: Should you add momentum?

Instead of equal-weighting the survivors, rank them by trailing 12-month return excluding the most recent month (called "12-1 momentum" -- the minus-one-month avoids short-term mean-reversion noise). Give the best 70%, the middle 20%, the worst 10%.

This is the only parameter we're adding: a momentum ranking that reshuffles the weights among funds that already passed the 200dma gate.

**The case for momentum (with cash exit):**

| | EW-cash | Momentum-cash | Delta |
|---|---|---|---|
| CAGR | 7.9% | 9.4% | +1.5% |
| Max Drawdown | -13.8% | -16.9% | -3.1% |
| Sharpe | 1.038 | 0.983 | -0.055 |
| $200K becomes | $821K | $1.06M | +$239K |

Full-period, momentum costs 0.06 Sharpe for +1.5% CAGR. Not a free lunch, but the CAGR pickup is meaningful.

But look at the windows:

| Start Date | EW-cash Sharpe | Momentum Sharpe | Winner |
|---|---|---|---|
| Full Period (2007) | 1.038 | 0.983 | Simple |
| Post-GFC (2009) | 1.011 | 1.014 | Momentum |
| All Tickers (2011) | 0.968 | 1.054 | Momentum |
| Pre-COVID (2018) | 0.976 | 1.089 | Momentum |
| Post-COVID (2021) | 0.917 | 1.276 | Momentum |

Momentum wins 4 out of 5 windows. Simple EW wins only the longest window (Full Period), while momentum wins every window starting 2009 or later — and the gap widens over time.

The cynic's read: this is GLD recency bias. Gold has gone on a historic tear, and momentum is just a fancy way of being overweight gold. If gold mean-reverts, momentum will underperform.

The bull's read: asset class divergence is increasing. In a world where stocks, bonds, and gold take turns leading, momentum captures whichever one is working. That's not a bug, it's the feature.

## Crisis Behavior

The 200dma filter earns its keep during crises. All four strategies were largely out of the market before the worst hits:

| Crisis | EW-cash | Momentum-cash | SPY B&H |
|---|---|---|---|
| Oct 2008 (GFC) | -0.6% | -0.6% | -16.5% |
| Mar 2020 (COVID) | +2.0% | +3.3% | -12.5% |
| 2022 Bear | -10.6% | -10.3% | -18.2% |
| 2025 Tariffs | +2.6% | +7.9% | -7.6% |

During the GFC, the filter had already moved to cash before the crash. During COVID, the strategies were positive while SPY lost 12.5%. During the 2025 tariff shock, momentum was up 7.9% (concentrated in gold) while SPY dropped 7.6%.

The 2022 bear is the hardest test -- stocks, bonds, and gold all declined together, breaking the usual pattern where at least one asset class rallies during a drawdown. Even so, the filter earned its keep: it exited TLT early in 2022 as the bond bear took hold, then exited SPY mid-year. The filtered strategies lost about half what SPY did. The filter worked; it was the cross-asset diversification that temporarily failed.

## What the Numbers Really Mean

The two regimes that truly test this strategy are the GFC and 2022 — the only two sustained multi-asset drawdowns in the 18-year sample. Everything else is either a short crash (COVID — V-shaped recovery too fast for monthly evaluation to matter much) or a calm bull market (2016–2019 — all assets above their 200dma, the filter never fires, every variant behaves identically).

The filter's value is almost entirely determined by how it performs during those two crises. The redistribution dial's behavior is driven by those same two periods. The momentum redistribution sweet spot discussed above? That's driven by just one of them (the GFC).

This is both the strength and the weakness of the backtest. Two genuine out-of-sample crises — the GFC happened after Faber published, and 2022 is a fundamentally different crisis type (rate-driven rather than credit-driven) — and the strategy handled both. But two data points is two data points. The next crisis could look different from both.

The strategy has been tested through two major crises of different types, and it worked in both. But the specific numbers (Sharpe, CAGR, MaxDD) are heavily influenced by those two periods. In a calm market, this strategy is just equal-weight buy-and-hold with a monthly check that never triggers.

## The Calm-Market Cost

Between the GFC and COVID (April 2009 – December 2019), the filter barely fires. All three assets spend most of the decade above their 200dma. The strategy is just equal-weight SPY/TLT/GLD with occasional false exits that create cash drag. Meanwhile:

| Portfolio | CAGR | Max Drawdown | Sharpe |
|---|---|---|---|
| SPY buy-and-hold | 16.0% | -19.4% | 1.06 |
| 60/40 (SPY/AGG) | 11.3% | -11.0% | 1.27 |
| All-Weather-Lite | 8.7% | -9.1% | 1.26 |
| **EW-cash (0%)** | **6.7%** | **-8.2%** | **1.00** |
| EW-renorm (100%) | 9.2% | -22.6% | 0.90 |

The strategy underperforms every benchmark on CAGR. At 0% redistribution, you're earning 6.7% while SPY does 16% and even conservative 60/40 does 11.3%. At 100% redistribution, you close the gap but take on -22.6% drawdown — worse than SPY.

This is the known cost of trend following. The 200dma filter's value comes entirely from drawdown protection during the GFC and 2022. In calm markets, it's a slight drag. The net result over the full period (including both crises) is still favorable on a risk-adjusted basis, but any window that excludes the crises will look bad.

The redistribution dial becomes more interesting in this light. At 0% redistribution you underperform benchmarks but with just -8.2% max drawdown — shallower than even All-Weather-Lite. At 100% you're closer to benchmark returns but with -22.6% max drawdown. The reader can see the tradeoff concretely in a period where the strategy was at a disadvantage.

Whether this tradeoff is worth it depends on whether you think another crisis is coming — and how much of your portfolio you're willing to lose when it does.

## What Didn't Work

We tested a lot of things that turned out not to matter:

**Inverse-volatility weighting**: Adds 0.028 Sharpe over equal weight. Not worth the complexity of computing rolling volatility.

**Concentrated momentum rotation (M1)**: Go 100% into the single best-momentum asset. Gets you 9.8% CAGR but with -33.6% max drawdown and a Sharpe of 0.64. Terrible risk-adjusted returns.

**Dual gate (200dma + positive momentum required)**: Only invest in assets that pass both the trend filter AND have positive trailing momentum. Gets you the shallowest drawdown ever tested (-9.6%) but sacrifices too much return (5.4% CAGR). You spend 48% of the time in cash.

**The 4th fund (managed futures/MF)**: Adding DBMF improved returns but introduced overfit concerns -- the fund has limited history and its addition felt like curve-fitting.

**Fixed 70/20/10 SPY-heavy with cash exit**: The worst of the 200dma strategies (0.865 Sharpe). When 70% of your portfolio is SPY and SPY drops below trend, you've already eaten most of the drawdown before the monthly signal fires.

## Current Positioning (as of Feb 2026)

All three funds are above their 200-day moving averages. 12-1 momentum readings:

- GLD: +64.2%
- SPY: +15.6%
- TLT: +5.5%

| Strategy | SPY | TLT | GLD | Cash |
|---|---|---|---|---|
| EW-cash | 33% | 33% | 33% | 0% |
| EW-renorm | 33% | 33% | 33% | 0% |
| Momentum-cash | 20% | 10% | 70% | 0% |
| Momentum-renorm | 20% | 10% | 70% | 0% |

With all three above trend, the cash exit vs renormalize distinction disappears -- both are fully invested. The only difference right now is whether you're equal weight or momentum-tilted toward gold.

## How to Choose

**EW-cash** if you want the simplest possible strategy. One rule (200dma), one action (go to cash), equal weights. Nothing to overfit, nothing to second-guess. Best for someone who wants to set it and forget it, check once a month, and sleep well knowing the max drawdown is around 14%.

**EW-renorm** if you want more growth and can handle the ride. Same zero-parameter simplicity but with renormalization. Your $200K becomes $1.47M instead of $821K over 18 years, but you'll see 22% drawdowns along the way. Best for someone with a long horizon who won't panic-sell during a drawdown.

**Momentum-cash** if you believe asset class divergence will persist and want to capture it. One extra step per month: rank the survivors by momentum and tilt 70/20/10. Wins in every window from 2009 forward. Best for someone who's willing to add a small amount of complexity for what has been a meaningful edge in recent years, while accepting that edge might not persist.

**Momentum-renorm** if you want maximum growth. 13.1% CAGR, $1.98M terminal value on $200K. But this is the most aggressive version -- concentrated bets, full investment, -21% drawdowns. Best for someone with a very long horizon, iron stomach, and conviction that momentum works.

Or blend them. Run EW-cash in one account and momentum in another. Use simple as your core and momentum as a satellite. Start simple and add momentum later if the regime persists.

## The Rules

Regardless of which cell you pick from the matrix, the mechanics are the same:

1. **Universe**: SPY, TLT, GLD
2. **Signal**: 200-day simple moving average, evaluated on the last trading day of each month
3. **Filter**: If a fund closes below its 200dma on month-end, it's "out" for the next month
4. **Weights**: Equal (33/33/33) or momentum-ranked (70/20/10 by 12-1 momentum)
5. **Exit mode**: Cash (reduce exposure) or renormalize (redistribute to survivors)
6. **Rebalance**: Monthly, on the last trading day

That's it. No optimization. No machine learning. No regime detection. Just a moving average and a monthly check.

---

*Fixed-rule backtest on SPY/TLT/GLD daily data from January 2007 to February 2026. No in-sample optimization, no parameter fitting -- all rules (200dma, equal weight, momentum lookback) are standard choices set before running. 2bps slippage per trade (negligible for monthly rebalancing of liquid ETFs at zero-commission brokers). $200,000 initial capital. Past performance does not guarantee future results.*
